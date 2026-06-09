from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from agents.base import AgentResult, Direction
from signal_generator import TradeSignal, align_trade_signal_direction, resolve_signal_direction
from strategy.near_tp1_breakeven import NearTp1BreakevenChecker, favorable_progress_r
from strategy.structure_weakness import (
    StructureWeaknessChecker,
    enrich_trade_entry_context,
    resolve_entry_rsi,
    resolve_entry_zone,
)
from strategy.trend_breakeven_alert import TrendBreakevenAlertChecker, sl_at_or_better_than_breakeven
from tracking.console import safe_print
from tracking.level_checks import stop_loss_hit, take_profit_hit
from tracking.profit_milestones import pending_profit_milestone_messages
from tracking.signal_csv import SignalCsvRow, SignalCsvStore
from tracking.trade_history import TradeHistoryStore, TradeRecord, TradeStatisticsCalculator, utc_now_iso

if TYPE_CHECKING:
    from runtime.m15_reversal_block import M15ReversalBlockGate
    from telegram.telegram_bot import TelegramBot

PriceFetcher = Callable[[str], float]
CandleFetcher = Callable[[str, str], dict[str, float]]
ContextFetcher = Callable[[str, str], dict]

DEFAULT_LOT_SIZE = 0.01


@dataclass
class ActiveTrade:
    symbol: str
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float
    reason: str
    open_time: str
    initial_stop_loss: float
    agents_agreement: str = "No"
    timeframe: str = ""
    entry_agent_results: dict[str, AgentResult] | None = None
    entry_trend_direction: Direction | None = None
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    entry_zone_kind: str | None = None
    entry_rsi: float | None = None
    last_rsi: float | None = None
    structure_warning_count: int = 0
    last_structure_candle_open_time: float | None = None
    last_structure_check_monotonic: float = 0.0
    telegram_message_id: int | None = None
    lot_size: float = DEFAULT_LOT_SIZE
    level1_warning_sent: bool = False
    level2_warning_sent: bool = False
    level2_streak: int = 0
    trend_warning_sent: bool = False
    last_trend_check_monotonic: float = 0.0
    last_trend_candle_open_time: float | None = None
    near_tp1_warning_sent: bool = False
    last_near_tp1_check_monotonic: float = 0.0
    last_near_tp1_candle_open_time: float | None = None
    peak_progress_r: float = 0.0
    tp1_reply_sent: bool = False
    tp2_reply_sent: bool = False
    tp3_reply_sent: bool = False
    sl_reply_sent: bool = False
    sl_proximity_warning_sent: bool = False
    profit_milestones_sent: list[int] | None = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    closed: bool = False
    close_time: str | None = None
    result: str | None = None
    recorded: bool = False

    @classmethod
    def from_signal(
        cls,
        symbol: str,
        signal: TradeSignal,
        *,
        agents_agreement: str = "No",
        timeframe: str = "",
        entry_agent_results: dict[str, AgentResult] | None = None,
        telegram_message_id: int | None = None,
        lot_size: float | None = None,
    ) -> ActiveTrade:
        aligned_signal = align_trade_signal_direction(signal)
        direction = resolve_signal_direction(aligned_signal)
        entry_trend = None
        if entry_agent_results and "trend_filter" in entry_agent_results:
            entry_trend = entry_agent_results["trend_filter"].direction

        return cls(
            symbol=symbol,
            direction=direction,
            entry=aligned_signal.entry,
            stop_loss=aligned_signal.stop_loss,
            initial_stop_loss=aligned_signal.stop_loss,
            tp1=aligned_signal.tp1,
            tp2=aligned_signal.tp2,
            tp3=aligned_signal.tp3,
            confidence=aligned_signal.confidence,
            reason=aligned_signal.reason,
            open_time=utc_now_iso(),
            agents_agreement=agents_agreement,
            timeframe=timeframe,
            entry_agent_results=entry_agent_results,
            entry_trend_direction=entry_trend,
            telegram_message_id=telegram_message_id,
            lot_size=aligned_signal.lot_size if lot_size is None else lot_size,
        )

    def profit_loss_r(self) -> float:
        risk = abs(self.entry - self.initial_stop_loss)
        if risk == 0 or self.result is None:
            return 0.0

        def signed_r(price: float) -> float:
            if self.direction == Direction.LONG:
                return (price - self.entry) / risk
            return (self.entry - price) / risk

        if not self.tp1_hit:
            exit_price = self.tp3 if self.result == "tp3" else self.stop_loss
            return signed_r(exit_price)

        # Partial management: 50% closed at TP1, 25% at TP2, 25% runner.
        total = 0.5 * signed_r(self.tp1)
        if self.tp2_hit:
            total += 0.25 * signed_r(self.tp2)
            final_exit = self.tp3 if self.result == "tp3" else self.stop_loss
            total += 0.25 * signed_r(final_exit)
        else:
            total += 0.5 * signed_r(self.stop_loss)
        return total

    def to_record(self) -> TradeRecord:
        return TradeRecord(
            symbol=self.symbol,
            direction=self.direction.value,
            entry=self.entry,
            stop_loss=self.stop_loss,
            tp1=self.tp1,
            tp2=self.tp2,
            tp3=self.tp3,
            confidence=self.confidence,
            reason=self.reason,
            open_time=self.open_time,
            close_time=self.close_time,
            result=self.result,
            tp1_hit=self.tp1_hit,
            tp2_hit=self.tp2_hit,
            tp3_hit=self.tp3_hit,
            entry_agent_results=self.entry_agent_results,
        )


class TradeMonitor:
    """Monitors live price for active signals and sends TP/SL updates."""

    def __init__(
        self,
        price_fetcher: PriceFetcher,
        history_store: TradeHistoryStore | None = None,
        stats_calculator: TradeStatisticsCalculator | None = None,
        telegram_bot: TelegramBot | None = None,
        signal_csv_store: SignalCsvStore | None = None,
        context_fetcher: ContextFetcher | None = None,
        structure_weakness_checker: StructureWeaknessChecker | None = None,
        trend_breakeven_checker: TrendBreakevenAlertChecker | None = None,
        near_tp1_breakeven_checker: NearTp1BreakevenChecker | None = None,
        m15_reversal_block: M15ReversalBlockGate | None = None,
        candle_fetcher: CandleFetcher | None = None,
    ) -> None:
        self.price_fetcher = price_fetcher
        self.candle_fetcher = candle_fetcher
        self.history_store = history_store or TradeHistoryStore()
        self.stats_calculator = stats_calculator or TradeStatisticsCalculator()
        self.telegram_bot = telegram_bot
        self.signal_csv_store = signal_csv_store or SignalCsvStore()
        self.context_fetcher = context_fetcher
        self.structure_weakness_checker = (
            structure_weakness_checker
            if structure_weakness_checker is not None
            else StructureWeaknessChecker(context_fetcher)
            if context_fetcher is not None
            else None
        )
        self.trend_breakeven_checker = (
            trend_breakeven_checker
            if trend_breakeven_checker is not None
            else TrendBreakevenAlertChecker(context_fetcher)
            if context_fetcher is not None
            else None
        )
        self.near_tp1_breakeven_checker = (
            near_tp1_breakeven_checker
            if near_tp1_breakeven_checker is not None
            else NearTp1BreakevenChecker(context_fetcher)
            if context_fetcher is not None
            else None
        )
        self.m15_reversal_block = m15_reversal_block
        from tracking.active_trades_store import ActiveTradesStore

        self.active_trades_store = ActiveTradesStore()
        self.active_trades = self.active_trades_store.load()

    def register_trade(
        self,
        trade: ActiveTrade,
        *,
        context: dict | None = None,
    ) -> ActiveTrade:
        if context is not None:
            enrich_trade_entry_context(trade, context)
        elif self.context_fetcher is not None and trade.entry_zone_low is None:
            try:
                timeframe = trade.timeframe or "15m"
                enrich_trade_entry_context(
                    trade,
                    self.context_fetcher(trade.symbol, timeframe),
                )
            except Exception as exc:
                safe_print(f"Entry context enrichment failed for {trade.symbol}: {exc}")

        if trade not in self.active_trades:
            self.active_trades.append(trade)
        self._persist_active_trades()
        return trade

    def _persist_active_trades(self) -> None:
        self.active_trades_store.save(self.active_trades)

    def track(
        self,
        trade: ActiveTrade,
        poll_interval: float = 5.0,
        max_polls: int | None = None,
    ) -> ActiveTrade:
        """Start monitoring an active trade until it closes."""
        self.register_trade(trade)
        return self.monitor(trade, poll_interval=poll_interval, max_polls=max_polls)

    def monitor(
        self,
        trade: ActiveTrade,
        poll_interval: float = 5.0,
        max_polls: int | None = None,
    ) -> ActiveTrade:
        self.monitor_all([trade], poll_interval=poll_interval, max_polls=max_polls)
        return trade

    def tick_all(self, trades: list[ActiveTrade] | None = None) -> list[ActiveTrade]:
        """Run one monitoring pass for open trades without blocking."""
        targets = trades if trades is not None else list(self.active_trades)
        closed_trades: list[ActiveTrade] = []

        for trade in targets:
            if trade.closed:
                continue

            candle = self._fetch_candle(trade)
            high = candle["high"]
            low = candle["low"]
            price = candle.get("close", (high + low) / 2)

            self._check_sl_proximity(trade, price)
            self._update_peak_progress(trade, high, low)
            if not trade.closed:
                self._check_trend_breakeven_alert(trade, price)
            if not trade.closed:
                self._check_near_tp1_breakeven_alert(trade, price)
            if not trade.closed:
                self._check_profit_milestones(trade, high, low)
            if not trade.closed:
                self._evaluate_candle(trade, high, low)
            if not trade.closed:
                self._check_structure_weakness(trade)

            if trade.closed:
                self._close_trade(trade)
                closed_trades.append(trade)
                if trade in self.active_trades:
                    self.active_trades.remove(trade)
                self._persist_active_trades()
            else:
                self._persist_active_trades()

        return closed_trades

    def monitor_all(
        self,
        trades: list[ActiveTrade] | None = None,
        poll_interval: float = 5.0,
        max_polls: int | None = None,
    ) -> list[ActiveTrade]:
        """Monitor multiple active trades in one polling loop."""
        targets = trades or [trade for trade in self.active_trades if not trade.closed]
        open_trades = [trade for trade in targets if not trade.closed]
        if not open_trades:
            return []

        safe_print()
        safe_print("=== TRADE MONITOR STARTED ===")
        for trade in open_trades:
            safe_print(f"Monitoring {trade.symbol} {trade.direction.value.upper()} trade...")

        polls = 0
        while open_trades:
            self.tick_all(open_trades)
            open_trades = [trade for trade in open_trades if not trade.closed]
            if not open_trades:
                break

            polls += 1
            if max_polls is not None and polls >= max_polls:
                safe_print()
                safe_print("Monitoring stopped (max polls reached). Open trades remain active.")
                break

            time.sleep(poll_interval)

        return targets

    def _check_sl_proximity(self, trade: ActiveTrade, price: float) -> None:
        if trade.sl_proximity_warning_sent or trade.closed:
            return

        original_risk = abs(trade.entry - trade.initial_stop_loss)
        if original_risk == 0:
            return

        if trade.direction == Direction.LONG:
            distance_to_sl = price - trade.stop_loss
        else:
            distance_to_sl = trade.stop_loss - price

        if distance_to_sl <= 0 or distance_to_sl > 0.30 * original_risk:
            return

        from tracking.trade_pnl import distance_to_sl_pips

        remaining_pips = distance_to_sl_pips(
            symbol=trade.symbol,
            direction=trade.direction.value,
            current_price=price,
            stop_loss=trade.stop_loss,
        )
        trade.sl_proximity_warning_sent = True
        self._notify_sl_proximity(trade, price, remaining_pips)
        self._persist_active_trades()

    def _notify_sl_proximity(
        self,
        trade: ActiveTrade,
        price: float,
        remaining_pips: float,
    ) -> None:
        from telegram.message_format import format_sl_proximity_warning

        message = format_sl_proximity_warning(
            current_price=price,
            remaining_pips=remaining_pips,
        )
        safe_print()
        safe_print(message)
        self._send_sl_proximity_warning(trade, price, remaining_pips)

    def _check_profit_milestones(
        self,
        trade: ActiveTrade,
        high: float,
        low: float,
    ) -> None:
        if stop_loss_hit(
            direction=trade.direction,
            high=high,
            low=low,
            stop_loss=trade.stop_loss,
        ):
            return

        if self._any_take_profit_hit_on_candle(trade, high, low):
            return

        messages = pending_profit_milestone_messages(trade, high=high, low=low)
        if not messages:
            return

        for message in messages:
            self._notify_profit_milestone(trade, message)
        self._persist_active_trades()

    def _any_take_profit_hit_on_candle(
        self,
        trade: ActiveTrade,
        high: float,
        low: float,
    ) -> bool:
        targets = (
            (trade.tp1, trade.tp1_hit),
            (trade.tp2, trade.tp2_hit),
            (trade.tp3, trade.tp3_hit),
        )
        for tp_price, already_hit in targets:
            if already_hit:
                continue
            if take_profit_hit(
                direction=trade.direction,
                high=high,
                low=low,
                tp_price=tp_price,
            ):
                return True
        return False

    def _notify_profit_milestone(self, trade: ActiveTrade, message: str) -> None:
        safe_print()
        safe_print(message)
        self._send_profit_milestone_reply(trade, message)

    def _send_profit_milestone_reply(self, trade: ActiveTrade, message: str) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_profit_milestone_reply(
                message,
                reply_to_message_id=trade.telegram_message_id,
            )
        except Exception as exc:
            safe_print(f"Telegram profit milestone failed: {exc}")

    def _update_peak_progress(self, trade: ActiveTrade, high: float, low: float) -> None:
        if trade.closed or trade.tp1_hit:
            return
        risk = abs(trade.entry - trade.initial_stop_loss)
        progress = favorable_progress_r(
            trade.direction,
            entry=trade.entry,
            risk=risk,
            high=high,
            low=low,
        )
        if progress > trade.peak_progress_r:
            trade.peak_progress_r = progress

    def _check_near_tp1_breakeven_alert(self, trade: ActiveTrade, price: float) -> None:
        if self.near_tp1_breakeven_checker is None:
            return

        try:
            assessment = self.near_tp1_breakeven_checker.analyze(
                trade,
                now_monotonic=time.monotonic(),
            )
        except Exception as exc:
            safe_print(f"Near-TP1 breakeven check failed for {trade.symbol}: {exc}")
            return

        if assessment is None or not assessment.should_move_sl_to_entry:
            return

        trade.near_tp1_warning_sent = True
        if self.m15_reversal_block is not None:
            self.m15_reversal_block.register_from_trade(trade)
        self._notify_near_tp1_breakeven_alert(trade, price, assessment.met_conditions)
        self._persist_active_trades()

    def _notify_near_tp1_breakeven_alert(
        self,
        trade: ActiveTrade,
        price: float,
        conditions: tuple[str, ...],
    ) -> None:
        safe_print()
        safe_print(
            f"Near-TP1 reversal for {trade.symbol} at {trade.peak_progress_r:.2f}R "
            f"— move SL to entry ({trade.entry:.2f})"
        )
        self._send_near_tp1_breakeven_alert(trade, price, conditions)

    def _send_near_tp1_breakeven_alert(
        self,
        trade: ActiveTrade,
        price: float,
        conditions: tuple[str, ...],
    ) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_near_tp1_breakeven_warning(
                reply_to_message_id=trade.telegram_message_id,
                open_time=trade.open_time,
                direction=trade.direction,
                current_price=price,
                entry=trade.entry,
                peak_progress_r=trade.peak_progress_r,
                conditions=conditions,
            )
        except Exception as exc:
            safe_print(f"Telegram near-TP1 breakeven alert failed: {exc}")

    def _check_trend_breakeven_alert(self, trade: ActiveTrade, price: float) -> None:
        if self.trend_breakeven_checker is None:
            return

        try:
            should_alert = self.trend_breakeven_checker.analyze(
                trade,
                now_monotonic=time.monotonic(),
            )
        except Exception as exc:
            safe_print(f"Trend breakeven check failed for {trade.symbol}: {exc}")
            return

        if not should_alert:
            return

        trade.trend_warning_sent = True
        self._notify_trend_breakeven_alert(trade, price)
        self._persist_active_trades()

    def _notify_trend_breakeven_alert(self, trade: ActiveTrade, price: float) -> None:
        safe_print()
        safe_print(
            f"Trend flip before TP1 for {trade.symbol} — move SL to entry ({trade.entry:.2f})"
        )
        self._send_trend_breakeven_alert(trade, price)

    def _send_trend_breakeven_alert(self, trade: ActiveTrade, price: float) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_trend_change_warning(
                reply_to_message_id=trade.telegram_message_id,
                open_time=trade.open_time,
                direction=trade.direction,
                current_price=price,
                entry=trade.entry,
            )
        except Exception as exc:
            safe_print(f"Telegram trend breakeven alert failed: {exc}")

    def _check_structure_weakness(self, trade: ActiveTrade) -> None:
        if self.structure_weakness_checker is None:
            return

        try:
            assessment = self.structure_weakness_checker.analyze(
                trade,
                now_monotonic=time.monotonic(),
            )
        except Exception as exc:
            safe_print(f"Structure weakness check failed for {trade.symbol}: {exc}")
            return

        if assessment is None or not assessment.should_warn or assessment.message is None:
            return

        trade.structure_warning_count += 1
        self._notify_structure_weakness(trade, assessment.message)
        self._persist_active_trades()

    def _notify_structure_weakness(self, trade: ActiveTrade, message: str) -> None:
        safe_print()
        safe_print(message)
        self._send_structure_weakness_warning(trade, message)

    def _fetch_candle(self, trade: ActiveTrade) -> dict[str, float]:
        timeframe = trade.timeframe or "15m"
        if self.candle_fetcher is not None:
            return self.candle_fetcher(trade.symbol, timeframe)

        price = self.price_fetcher(trade.symbol)
        return {"high": price, "low": price, "close": price}

    def _evaluate_candle(self, trade: ActiveTrade, high: float, low: float) -> None:
        if trade.direction == Direction.LONG:
            self._evaluate_long(trade, high, low)
        else:
            self._evaluate_short(trade, high, low)

    def _evaluate_long(self, trade: ActiveTrade, high: float, low: float) -> None:
        if stop_loss_hit(
            direction=Direction.LONG,
            high=high,
            low=low,
            stop_loss=trade.stop_loss,
        ):
            self._finalize_stop_loss(trade)
            return

        if not trade.tp1_hit and take_profit_hit(
            direction=Direction.LONG,
            high=high,
            low=low,
            tp_price=trade.tp1,
        ):
            trade.tp1_hit = True
            trade.stop_loss = trade.entry
            self._notify(trade, "tp1")

        if trade.closed:
            return

        if not trade.tp2_hit and take_profit_hit(
            direction=Direction.LONG,
            high=high,
            low=low,
            tp_price=trade.tp2,
        ):
            trade.tp2_hit = True
            trade.stop_loss = trade.tp1
            self._notify(trade, "tp2")

        if trade.closed:
            return

        if take_profit_hit(
            direction=Direction.LONG,
            high=high,
            low=low,
            tp_price=trade.tp3,
        ):
            trade.tp3_hit = True
            self._finalize(trade, "tp3")

    def _evaluate_short(self, trade: ActiveTrade, high: float, low: float) -> None:
        if stop_loss_hit(
            direction=Direction.SHORT,
            high=high,
            low=low,
            stop_loss=trade.stop_loss,
        ):
            self._finalize_stop_loss(trade)
            return

        if not trade.tp1_hit and take_profit_hit(
            direction=Direction.SHORT,
            high=high,
            low=low,
            tp_price=trade.tp1,
        ):
            trade.tp1_hit = True
            trade.stop_loss = trade.entry
            self._notify(trade, "tp1")

        if trade.closed:
            return

        if not trade.tp2_hit and take_profit_hit(
            direction=Direction.SHORT,
            high=high,
            low=low,
            tp_price=trade.tp2,
        ):
            trade.tp2_hit = True
            trade.stop_loss = trade.tp1
            self._notify(trade, "tp2")

        if trade.closed:
            return

        if take_profit_hit(
            direction=Direction.SHORT,
            high=high,
            low=low,
            tp_price=trade.tp3,
        ):
            trade.tp3_hit = True
            self._finalize(trade, "tp3")

    def _notify(self, trade: ActiveTrade, event: str) -> None:
        from telegram.message_format import format_trade_result

        message = format_trade_result(trade.symbol, trade.direction, event)
        safe_print()
        safe_print(message)
        self._send_telegram(trade, event)

    def _finalize_stop_loss(self, trade: ActiveTrade) -> None:
        from strategy.trend_breakeven_alert import sl_at_or_better_than_breakeven

        if sl_at_or_better_than_breakeven(trade):
            self._finalize(trade, "breakeven")
            return
        self._finalize(trade, "stop_loss")

    def _finalize(self, trade: ActiveTrade, result: str) -> None:
        trade.closed = True
        trade.result = result
        trade.close_time = utc_now_iso()
        if self.m15_reversal_block is not None and (
            trade.near_tp1_warning_sent or (result == "breakeven" and not trade.tp1_hit)
        ):
            self.m15_reversal_block.register_from_trade(trade)
        self._notify(trade, result)

    def _send_telegram(self, trade: ActiveTrade, event: str) -> None:
        if self.telegram_bot is None:
            return

        reply_to = trade.telegram_message_id
        try:
            if event == "stop_loss":
                if trade.sl_reply_sent:
                    return
                self.telegram_bot.send_stop_loss_reply(
                    trade,
                    reply_to_message_id=reply_to,
                )
                trade.sl_reply_sent = True
                return

            if event == "breakeven":
                if trade.sl_reply_sent:
                    return
                if trade.tp1_hit:
                    self.telegram_bot.send_post_tp_close_reply(
                        trade,
                        reply_to_message_id=reply_to,
                    )
                else:
                    self.telegram_bot.send_breakeven_reply(
                        trade,
                        reply_to_message_id=reply_to,
                    )
                trade.sl_reply_sent = True
                return

            tp_map = {
                "tp1": (1, trade.tp1, "tp1_reply_sent"),
                "tp2": (2, trade.tp2, "tp2_reply_sent"),
                "tp3": (3, trade.tp3, "tp3_reply_sent"),
            }
            if event not in tp_map:
                return

            level, tp_price, sent_flag = tp_map[event]
            if getattr(trade, sent_flag):
                return

            self.telegram_bot.send_take_profit_reply(
                trade,
                tp_level=level,
                tp_price=tp_price,
                reply_to_message_id=reply_to,
            )
            setattr(trade, sent_flag, True)
        except Exception as exc:
            safe_print(f"Telegram update failed: {exc}")

    def _send_structure_weakness_warning(
        self,
        trade: ActiveTrade,
        message: str,
    ) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_structure_weakness_warning(
                message,
                reply_to_message_id=trade.telegram_message_id,
            )
        except Exception as exc:
            safe_print(f"Telegram structure weakness warning failed: {exc}")

    def _send_sl_proximity_warning(
        self,
        trade: ActiveTrade,
        price: float,
        remaining_pips: float,
    ) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_sl_proximity_warning(
                reply_to_message_id=trade.telegram_message_id,
                current_price=price,
                remaining_pips=remaining_pips,
            )
        except Exception as exc:
            safe_print(f"Telegram SL proximity warning failed: {exc}")

    def _close_trade(self, trade: ActiveTrade) -> None:
        if trade.recorded:
            return

        trade.recorded = True
        self.history_store.add_trade(trade.to_record())
        if trade.closed and trade.result is not None:
            self.signal_csv_store.append(
                SignalCsvRow(
                    date=trade.open_time,
                    symbol=trade.symbol,
                    direction=trade.direction.value,
                    entry=trade.entry,
                    sl=trade.initial_stop_loss,
                    tp1=trade.tp1,
                    tp2=trade.tp2,
                    tp3=trade.tp3,
                    result=trade.result,
                    profit_loss=trade.profit_loss_r(),
                    confidence=trade.confidence,
                    agents_agreement=trade.agents_agreement,
                )
            )
        self.stats_calculator.print_statistics(store=self.history_store)
