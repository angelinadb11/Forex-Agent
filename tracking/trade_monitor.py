from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from agents.base import AgentResult, Direction
from signal_generator import TradeSignal, align_trade_signal_direction, resolve_signal_direction
from strategy.trade_update import LEVEL2_STANDARD_CONFIRMATION_CYCLES, TradeUpdateChecker, WarningLevel
from tracking.console import safe_print
from tracking.signal_csv import SignalCsvRow, SignalCsvStore
from tracking.trade_history import TradeHistoryStore, TradeRecord, TradeStatisticsCalculator, utc_now_iso

if TYPE_CHECKING:
    from telegram.telegram_bot import TelegramBot

PriceFetcher = Callable[[str], float]
ContextFetcher = Callable[[str, str], dict]


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
    level1_warning_sent: bool = False
    level2_warning_sent: bool = False
    level2_streak: int = 0
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
    ) -> ActiveTrade:
        aligned_signal = align_trade_signal_direction(signal)
        direction = resolve_signal_direction(aligned_signal)
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
        )

    def profit_loss_r(self) -> float:
        risk = abs(self.entry - self.initial_stop_loss)
        if risk == 0 or self.result is None:
            return 0.0

        if self.result == "tp3":
            exit_price = self.tp3
        else:
            exit_price = self.stop_loss

        if self.direction == Direction.LONG:
            return (exit_price - self.entry) / risk
        return (self.entry - exit_price) / risk

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
        trade_update_checker: TradeUpdateChecker | None = None,
    ) -> None:
        self.price_fetcher = price_fetcher
        self.history_store = history_store or TradeHistoryStore()
        self.stats_calculator = stats_calculator or TradeStatisticsCalculator()
        self.telegram_bot = telegram_bot
        self.signal_csv_store = signal_csv_store or SignalCsvStore()
        self.context_fetcher = context_fetcher
        self.trade_update_checker = (
            trade_update_checker
            if trade_update_checker is not None
            else TradeUpdateChecker(context_fetcher)
            if context_fetcher is not None
            else None
        )
        self.active_trades: list[ActiveTrade] = []

    def track(
        self,
        trade: ActiveTrade,
        poll_interval: float = 5.0,
        max_polls: int | None = None,
    ) -> ActiveTrade:
        """Start monitoring an active trade until it closes."""
        self.active_trades.append(trade)
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

            price = self.price_fetcher(trade.symbol)
            self._evaluate_price(trade, price)
            if not trade.closed:
                self._check_trade_update(trade)

            if trade.closed:
                self._close_trade(trade)
                closed_trades.append(trade)
                if trade in self.active_trades:
                    self.active_trades.remove(trade)

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

    def _check_trade_update(self, trade: ActiveTrade) -> None:
        if self.trade_update_checker is None:
            return

        try:
            assessment = self.trade_update_checker.analyze(trade)
        except Exception as exc:
            safe_print(f"Trade update check failed for {trade.symbol}: {exc}")
            return

        if assessment is None:
            return

        if assessment.level2_instant:
            trade.level2_streak = 0
            if not trade.level2_warning_sent:
                self._notify_high_risk_update(trade, list(assessment.level2_reasons))
                trade.level2_warning_sent = True
            return

        if assessment.level2_standard:
            trade.level2_streak += 1
            if (
                trade.level2_streak >= LEVEL2_STANDARD_CONFIRMATION_CYCLES
                and not trade.level2_warning_sent
            ):
                self._notify_high_risk_update(trade, list(assessment.level2_reasons))
                trade.level2_warning_sent = True
            return

        trade.level2_streak = 0

        if assessment.level != WarningLevel.LEVEL_1:
            return
        if trade.level1_warning_sent or trade.level2_warning_sent:
            return

        trade.level1_warning_sent = True
        self._notify_trade_update_level1(trade, list(assessment.reasons))

    def _notify_trade_update_level1(self, trade: ActiveTrade, reasons: list[str]) -> None:
        from telegram.message_format import format_trade_update_warning

        message = format_trade_update_warning(trade.symbol, trade.direction, reasons)
        safe_print()
        safe_print(message)
        self._send_trade_update_level1(trade, reasons)

    def _notify_high_risk_update(self, trade: ActiveTrade, reasons: list[str]) -> None:
        from telegram.message_format import format_high_risk_update

        message = format_high_risk_update(trade.symbol, trade.direction, reasons)
        safe_print()
        safe_print(message)
        self._send_high_risk_update(trade, reasons)

    def _evaluate_price(self, trade: ActiveTrade, price: float) -> None:
        if trade.direction == Direction.LONG:
            self._evaluate_long(trade, price)
        else:
            self._evaluate_short(trade, price)

    def _evaluate_long(self, trade: ActiveTrade, price: float) -> None:
        if price <= trade.stop_loss:
            self._finalize_stop_loss(trade)
            return

        if not trade.tp1_hit and price >= trade.tp1:
            trade.tp1_hit = True
            trade.stop_loss = trade.entry
            self._notify(trade, "tp1")

        if trade.closed:
            return

        if not trade.tp2_hit and price >= trade.tp2:
            trade.tp2_hit = True
            trade.stop_loss = trade.tp1
            self._notify(trade, "tp2")

        if trade.closed:
            return

        if price >= trade.tp3:
            trade.tp3_hit = True
            self._finalize(trade, "tp3")

    def _evaluate_short(self, trade: ActiveTrade, price: float) -> None:
        if price >= trade.stop_loss:
            self._finalize_stop_loss(trade)
            return

        if not trade.tp1_hit and price <= trade.tp1:
            trade.tp1_hit = True
            trade.stop_loss = trade.entry
            self._notify(trade, "tp1")

        if trade.closed:
            return

        if not trade.tp2_hit and price <= trade.tp2:
            trade.tp2_hit = True
            trade.stop_loss = trade.tp1
            self._notify(trade, "tp2")

        if trade.closed:
            return

        if price <= trade.tp3:
            trade.tp3_hit = True
            self._finalize(trade, "tp3")

    def _notify(self, trade: ActiveTrade, event: str) -> None:
        from telegram.message_format import format_trade_result

        message = format_trade_result(trade.symbol, trade.direction, event)
        safe_print()
        safe_print(message)
        self._send_telegram(trade, event)

    def _finalize_stop_loss(self, trade: ActiveTrade) -> None:
        self._finalize(trade, "stop_loss")

    def _finalize(self, trade: ActiveTrade, result: str) -> None:
        trade.closed = True
        trade.result = result
        trade.close_time = utc_now_iso()
        self._notify(trade, result)

    def _send_telegram(self, trade: ActiveTrade, event: str) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_trade_update(trade.symbol, trade.direction, event)
        except Exception as exc:
            safe_print(f"Telegram update failed: {exc}")

    def _send_trade_update_level1(self, trade: ActiveTrade, reasons: list[str]) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_trade_update_warning(
                trade.symbol,
                trade.direction,
                reasons,
            )
        except Exception as exc:
            safe_print(f"Telegram trade update failed: {exc}")

    def _send_high_risk_update(self, trade: ActiveTrade, reasons: list[str]) -> None:
        if self.telegram_bot is None:
            return

        try:
            self.telegram_bot.send_high_risk_update(
                trade.symbol,
                trade.direction,
                reasons,
            )
        except Exception as exc:
            safe_print(f"Telegram high-risk update failed: {exc}")

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
