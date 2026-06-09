from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from agents.base import Direction
from signal_generator import TradeSignal, resolve_signal_direction
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
    ) -> SimulatedTradeResult | None:
        if mode == TradeManagementMode.LEGACY:
            return self._simulate_legacy(signal, future_candles, entry_index)
        return self._simulate_partial(signal, future_candles, entry_index)

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
                    sl_r = (stop_loss - signal.entry) / risk
                    pnl_r = cumulative_r + position_remaining * sl_r
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
                        TradeManagementMode.PARTIAL,
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
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        cumulative_r,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.PARTIAL,
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
                        TradeManagementMode.PARTIAL,
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
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        cumulative_r,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                        TradeManagementMode.PARTIAL,
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
    ) -> SimulatedTradeResult:
        win = pnl_r > 0 or tp1_hit or result == "tp3"
        loss = pnl_r < 0 and result == "stop_loss" and not tp1_hit

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
        )
