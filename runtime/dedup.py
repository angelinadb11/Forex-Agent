from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config.symbols import resolve_symbol
from signal_generator import TradeSignal, resolve_signal_direction


@dataclass(frozen=True)
class DedupDecision:
    allowed: bool
    reason: str | None = None


class SignalDedupGate:
    """Prevent duplicate signals and multiple open trades per symbol."""

    def __init__(
        self,
        *,
        duplicate_entry_tolerance_pct: float = 0.001,
        signal_cooldown_minutes: int = 60,
    ) -> None:
        self.duplicate_entry_tolerance_pct = duplicate_entry_tolerance_pct
        self.signal_cooldown_minutes = signal_cooldown_minutes
        self._last_fingerprints: dict[str, str] = {}
        self._cooldown_until: dict[str, datetime] = {}

    def can_publish(
        self,
        symbol: str,
        signal: TradeSignal,
        open_symbols: set[str],
    ) -> DedupDecision:
        display_symbol = resolve_symbol(symbol).display

        if display_symbol in open_symbols:
            return DedupDecision(
                allowed=False,
                reason=f"open trade already active for {display_symbol}",
            )

        fingerprint = self._fingerprint(signal)
        if self._last_fingerprints.get(display_symbol) == fingerprint:
            return DedupDecision(
                allowed=False,
                reason=f"duplicate setup for {display_symbol}",
            )

        cooldown_until = self._cooldown_until.get(display_symbol)
        if cooldown_until is not None:
            now = datetime.now(timezone.utc)
            if now < cooldown_until:
                return DedupDecision(
                    allowed=False,
                    reason=f"signal cooldown active for {display_symbol}",
                )

        return DedupDecision(allowed=True)

    def record_published(self, symbol: str, signal: TradeSignal) -> None:
        display_symbol = resolve_symbol(symbol).display
        self._last_fingerprints[display_symbol] = self._fingerprint(signal)
        if self.signal_cooldown_minutes > 0:
            self._cooldown_until[display_symbol] = datetime.now(timezone.utc) + timedelta(
                minutes=self.signal_cooldown_minutes
            )

    def _fingerprint(self, signal: TradeSignal) -> str:
        direction = resolve_signal_direction(signal)
        entry = self._round_level(signal.entry)
        stop_loss = self._round_level(signal.stop_loss)
        return f"{direction.value}|{entry}|{stop_loss}"

    def _round_level(self, price: float) -> str:
        tolerance = max(price * self.duplicate_entry_tolerance_pct, 1e-8)
        rounded = round(price / tolerance) * tolerance
        return f"{rounded:.8f}"
