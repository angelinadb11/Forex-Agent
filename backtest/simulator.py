from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agents.base import Direction
from signal_generator import TradeSignal, resolve_signal_direction


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradeSimulator:
    """Simulates TP/SL management on future OHLC candles."""

    def simulate(
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
                if low <= stop_loss:
                    pnl_r = (stop_loss - signal.entry) / risk
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        stop_loss,
                        pnl_r,
                        "stop_loss",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                    )

                if not tp1_hit and high >= signal.tp1:
                    tp1_hit = True
                    stop_loss = signal.entry

                if not tp2_hit and high >= signal.tp2:
                    tp2_hit = True
                    stop_loss = signal.tp1

                if high >= signal.tp3:
                    tp3_hit = True
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        3.0,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                    )

            else:
                if high >= stop_loss:
                    pnl_r = (signal.entry - stop_loss) / risk
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        stop_loss,
                        pnl_r,
                        "stop_loss",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
                    )

                if not tp1_hit and low <= signal.tp1:
                    tp1_hit = True
                    stop_loss = signal.entry

                if not tp2_hit and low <= signal.tp2:
                    tp2_hit = True
                    stop_loss = signal.tp1

                if low <= signal.tp3:
                    tp3_hit = True
                    return self._build_result(
                        signal,
                        entry_index,
                        exit_index,
                        signal.tp3,
                        3.0,
                        "tp3",
                        tp1_hit,
                        tp2_hit,
                        tp3_hit,
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
    ) -> SimulatedTradeResult:
        win = tp1_hit or result == "tp3"
        loss = result == "stop_loss" and not tp1_hit

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
        )
