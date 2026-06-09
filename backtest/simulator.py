from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from agents.base import Direction
from signal_generator import TradeSignal, resolve_signal_direction
from strategy.near_tp1_breakeven import (
    assess_near_tp1_reversal,
    favorable_progress_r,
)
from strategy.structure_weakness import resolve_entry_rsi, resolve_entry_zone
from tracking.level_checks import stop_loss_hit, take_profit_hit

TP1_CLOSE_FRACTION = 0.50
TP2_CLOSE_FRACTION = 0.25
TP3_CLOSE_FRACTION = 0.25
BREAKEVEN_TOLERANCE_RATIO = 1e-6


def is_breakeven_stop_level(entry: float, stop_loss: float, risk: float) -> bool:
    tolerance = max(1e-9, risk * BREAKEVEN_TOLERANCE_RATIO)
    return abs(stop_loss - entry) <= tolerance


def resolve_stop_exit_result(
    *,
    entry: float,
    stop_loss: float,
    risk: float,
) -> str:
    if is_breakeven_stop_level(entry, stop_loss, risk):
        return "breakeven"
    return "stop_loss"


class TradeManagementMode(str, Enum):
    LEGACY = "legacy"
    PARTIAL = "partial"
    PARTIAL_NEAR_TP1_BE = "partial_near_tp1_be"


@dataclass
class SimulatedTradeResult:
    entry_index: int
    exit_index: int
    direction: str
    entry: float
    exit_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk: float
    pnl_r: float
    result: str
    win: bool
    loss: bool
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    confidence: float
    reason: str
    management_mode: str = TradeManagementMode.PARTIAL.value
    near_tp1_be_triggered: bool = False
    entry_zone_low: float | None = None
    entry_zone_high: float | None = None
    entry_rsi: float | None = None
    last_rsi: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradeSimulator:
    """Simulates TP/SL management on future OHLC candles."""

    def simulate(
        self,
        signal: TradeSignal,
        future_candles: list[dict[str, float]],
        entry_index: int,
        *,
        mode: TradeManagementMode = TradeManagementMode.PARTIAL,
        all_candles: list[dict[str, float]] | None = None,
        zone_catalog=None,
        symbol: str = "XAUUSD",
    ) -> SimulatedTradeResult | None:
        if mode == TradeManagementMode.LEGACY:
            return self._simulate_legacy(signal, future_candles, entry_index)
        enable_near_tp1_be = mode == TradeManagementMode.PARTIAL_NEAR_TP1_BE
        return self._simulate_partial(
            signal,
            future_candles,
            entry_index,
            enable_near_tp1_be=enable_near_tp1_be,
            all_candles=all_candles,
            zone_catalog=zone_catalog,
            symbol=symbol,
            management_mode=mode,
        )

    def _simulate_legacy(
        self,
        signal: TradeSignal,
        future_candles: list[dict[str, float]],
        entry_index: int,
    ) -> SimulatedTradeResult | None:
        if not future_candles:
            return None

        stop_loss = signal.stop_loss
        risk = abs(signal.entry - signal.stop_loss)
        if risk == 0:
            return None

        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        trade_direction = resolve_signal_direction(signal)

        for offset, candle in enumerate(future_candles):
            exit_index = entry_index + offset + 1
            high = candle["high"]
            low = candle["low"]

            if trade_direction == Direction.LONG:
                if stop_loss_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    stop_loss=stop_loss,
                ):
                    pnl_r = (stop_loss - signal.entry) / risk
                    exit_result = resolve_stop_exit_result(
                        entry=signal.entry,
                        stop_loss=stop_loss,
                        risk=risk,
                    )
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        stop_loss,
                        pnl_r,
                        exit_result,
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.LEGACY,
                    )

                if not tp1_hit and take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp1,
                ):
                    tp1_hit = True
                    stop_loss = signal.entry

                if not tp2_hit and take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp2,
                ):
                    tp2_hit = True
                    stop_loss = signal.tp1

                if take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp3,
                ):
                    tp3_hit = True
                    pnl_r = (signal.tp3 - signal.entry) / risk
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        pnl_r,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.LEGACY,
                    )

            else:
                if stop_loss_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    stop_loss=stop_loss,
                ):
                    pnl_r = (signal.entry - stop_loss) / risk
                    exit_result = resolve_stop_exit_result(
                        entry=signal.entry,
                        stop_loss=stop_loss,
                        risk=risk,
                    )
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        stop_loss,
                        pnl_r,
                        exit_result,
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.LEGACY,
                    )

                if not tp1_hit and take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp1,
                ):
                    tp1_hit = True
                    stop_loss = signal.entry

                if not tp2_hit and take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp2,
                ):
                    tp2_hit = True
                    stop_loss = signal.tp1

                if take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp3,
                ):
                    tp3_hit = True
                    pnl_r = (signal.entry - signal.tp3) / risk
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        pnl_r,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.LEGACY,
                    )

        return None

    def _simulate_partial(
        self,
        signal: TradeSignal,
        future_candles: list[dict[str, float]],
        entry_index: int,
        *,
        enable_near_tp1_be: bool = False,
        all_candles: list[dict[str, float]] | None = None,
        zone_catalog=None,
        symbol: str = "XAUUSD",
        management_mode: TradeManagementMode = TradeManagementMode.PARTIAL,
    ) -> SimulatedTradeResult | None:
        if not future_candles:
            return None

        stop_loss = signal.stop_loss
        risk = abs(signal.entry - signal.stop_loss)
        if risk == 0:
            return None

        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        position_remaining = 1.0
        cumulative_r = 0.0
        trade_direction = resolve_signal_direction(signal)
        peak_progress_r = 0.0
        near_tp1_be_triggered = False
        entry_zone_low: float | None = None
        entry_zone_high: float | None = None
        entry_rsi: float | None = None
        last_rsi: float | None = None

        if enable_near_tp1_be and all_candles is not None:
            entry_context: dict = {
                "symbol": symbol,
                "candles": all_candles[: entry_index + 1],
                "bar_index": entry_index,
            }
            if zone_catalog is not None:
                entry_context["zone_catalog"] = zone_catalog
            zone = resolve_entry_zone(entry_context, trade_direction, signal.entry)
            if zone is not None:
                entry_zone_low = zone.zone_low
                entry_zone_high = zone.zone_high
            entry_rsi = resolve_entry_rsi(entry_context)
            last_rsi = entry_rsi

        def finish_partial_result(
            exit_index: int,
            exit_price: float,
            pnl_r: float,
            result: str,
        ) -> SimulatedTradeResult:
            return self._build_result(
                signal,
                entry_index,
                exit_index,
                exit_price,
                pnl_r,
                result,
                tp1_hit,
                tp2_hit,
                tp3_hit,
                management_mode,
                near_tp1_be_triggered=near_tp1_be_triggered,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                entry_rsi=entry_rsi,
                last_rsi=last_rsi,
            )

        for offset, candle in enumerate(future_candles):
            exit_index = entry_index + offset + 1
            high = candle["high"]
            low = candle["low"]

            peak_progress_r = max(
                peak_progress_r,
                favorable_progress_r(
                    trade_direction,
                    entry=signal.entry,
                    risk=risk,
                    high=high,
                    low=low,
                ),
            )

            if enable_near_tp1_be and all_candles is not None and not tp1_hit:
                sl_at_be = is_breakeven_stop_level(signal.entry, stop_loss, risk)
                if not sl_at_be:
                    current_context = {
                        "symbol": symbol,
                        "candles": all_candles[: exit_index + 1],
                        "bar_index": exit_index,
                    }
                    if zone_catalog is not None:
                        current_context["zone_catalog"] = zone_catalog
                    assessment = assess_near_tp1_reversal(
                        trade_direction,
                        peak_progress_r=peak_progress_r,
                        tp1_hit=tp1_hit,
                        sl_at_breakeven=sl_at_be,
                        m15_context=current_context,
                        entry_zone_low=entry_zone_low,
                        entry_zone_high=entry_zone_high,
                        entry_rsi=entry_rsi,
                        previous_rsi=last_rsi,
                    )
                    current_rsi = resolve_entry_rsi(current_context)
                    if current_rsi is not None:
                        last_rsi = current_rsi
                    if assessment.should_move_sl_to_entry:
                        near_tp1_be_triggered = True
                        stop_loss = signal.entry

            if trade_direction == Direction.LONG:
                if stop_loss_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    stop_loss=stop_loss,
                ):
                    sl_r = (stop_loss - signal.entry) / risk
                    pnl_r = cumulative_r + position_remaining * sl_r
                    exit_result = resolve_stop_exit_result(
                        entry=signal.entry,
                        stop_loss=stop_loss,
                        risk=risk,
                    )
                    return finish_partial_result(
                        exit_index,
                        stop_loss,
                        pnl_r,
                        exit_result,
                    )

                if not tp1_hit and take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp1,
                ):
                    tp1_hit = True
                    cumulative_r += TP1_CLOSE_FRACTION * (
                        (signal.tp1 - signal.entry) / risk
                    )
                    position_remaining -= TP1_CLOSE_FRACTION
                    stop_loss = signal.entry

                if not tp2_hit and take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp2,
                ):
                    tp2_hit = True
                    cumulative_r += TP2_CLOSE_FRACTION * (
                        (signal.tp2 - signal.entry) / risk
                    )
                    position_remaining -= TP2_CLOSE_FRACTION
                    stop_loss = signal.tp1

                if take_profit_hit(
                    direction=Direction.LONG,
                    high=high,
                    low=low,
                    tp_price=signal.tp3,
                ):
                    tp3_hit = True
                    cumulative_r += TP3_CLOSE_FRACTION * (
                        (signal.tp3 - signal.entry) / risk
                    )
                    return finish_partial_result(
                        exit_index,
                        signal.tp3,
                        cumulative_r,
                        "tp3",
                    )

            else:
                if stop_loss_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    stop_loss=stop_loss,
                ):
                    sl_r = (signal.entry - stop_loss) / risk
                    pnl_r = cumulative_r + position_remaining * sl_r
                    exit_result = resolve_stop_exit_result(
                        entry=signal.entry,
                        stop_loss=stop_loss,
                        risk=risk,
                    )
                    return finish_partial_result(
                        exit_index,
                        stop_loss,
                        pnl_r,
                        exit_result,
                    )

                if not tp1_hit and take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp1,
                ):
                    tp1_hit = True
                    cumulative_r += TP1_CLOSE_FRACTION * (
                        (signal.entry - signal.tp1) / risk
                    )
                    position_remaining -= TP1_CLOSE_FRACTION
                    stop_loss = signal.entry

                if not tp2_hit and take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp2,
                ):
                    tp2_hit = True
                    cumulative_r += TP2_CLOSE_FRACTION * (
                        (signal.entry - signal.tp2) / risk
                    )
                    position_remaining -= TP2_CLOSE_FRACTION
                    stop_loss = signal.tp1

                if take_profit_hit(
                    direction=Direction.SHORT,
                    high=high,
                    low=low,
                    tp_price=signal.tp3,
                ):
                    tp3_hit = True
                    cumulative_r += TP3_CLOSE_FRACTION * (
                        (signal.entry - signal.tp3) / risk
                    )
                    return finish_partial_result(
                        exit_index,
                        signal.tp3,
                        cumulative_r,
                        "tp3",
                    )

        return None

    def _build_result(
        self,
        signal: TradeSignal,
        entry_index: int,
        exit_index: int,
        exit_price: float,
        pnl_r: float,
        result: str,
        tp1_hit: bool,
        tp2_hit: bool,
        tp3_hit: bool,
        mode: TradeManagementMode,
        *,
        near_tp1_be_triggered: bool = False,
        entry_zone_low: float | None = None,
        entry_zone_high: float | None = None,
        entry_rsi: float | None = None,
        last_rsi: float | None = None,
    ) -> SimulatedTradeResult:
        win = pnl_r > 0 or tp1_hit or result == "tp3"
        from tracking.trade_outcome import is_full_stop_loss_from_values

        loss = is_full_stop_loss_from_values(
            result=result,
            tp1_hit=tp1_hit,
            pnl_r=pnl_r,
            exit_price=exit_price,
            entry=signal.entry,
            initial_stop_loss=signal.stop_loss,
            risk=abs(signal.entry - signal.stop_loss),
        )

        return SimulatedTradeResult(
            entry_index=entry_index,
            exit_index=exit_index,
            direction=resolve_signal_direction(signal).value,
            entry=signal.entry,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            risk=abs(signal.entry - signal.stop_loss),
            pnl_r=pnl_r,
            result=result,
            win=win,
            loss=loss,
            tp1_hit=tp1_hit,
            tp2_hit=tp2_hit,
            tp3_hit=tp3_hit,
            confidence=signal.confidence,
            reason=signal.reason,
            management_mode=mode.value,
            near_tp1_be_triggered=near_tp1_be_triggered,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            entry_rsi=entry_rsi,
            last_rsi=last_rsi,
        )
