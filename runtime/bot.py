from __future__ import annotations

import logging
import time
from typing import Callable

from agents.base import AgentResult
from config.symbols import resolve_symbol
from data import MarketDataProvider
from runtime.dedup import SignalDedupGate
from signal_generator import SignalGenerator, TradeSignal
from strategy import format_agents_agreement
from runtime.m15_reversal_block import M15ReversalBlockGate
from strategy.signal_filter import FilterResult, SignalFilter
from tracking.trade_monitor import ActiveTrade, TradeMonitor
from tracking.console import safe_print

from strategy.scalp_mode import SCALP_TIMEFRAME, is_scalp_enabled
from strategy.sweep_fvg_scalp import SWEEP_FVG_TIMEFRAME, is_sweep_fvg_scalp_enabled
from strategy.turtle_soup_scalp import TURTLE_SOUP_TIMEFRAME, is_turtle_soup_scalp_enabled

AnalyzeSymbolFn = Callable[
    ...,
    tuple[TradeSignal | None, dict[str, AgentResult] | None, FilterResult | None, dict | None],
]
AnalyzeScalpFn = Callable[..., tuple[TradeSignal | None, dict | None, object]]
PublishScalpFn = Callable[..., ActiveTrade]

DEFAULT_SCALP_SCAN_INTERVAL_SECONDS = 60.0
DEFAULT_PREMIUM_SCAN_INTERVAL_SECONDS = 60.0
DEFAULT_TURTLE_SCAN_INTERVAL_SECONDS = 60.0


class BotRuntime:
    """Continuous or one-shot orchestration for scanning and trade monitoring."""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        timeframe: str,
        logger: logging.Logger,
        provider: MarketDataProvider,
        signal_filter: SignalFilter,
        signal_generator: SignalGenerator,
        monitor: TradeMonitor,
        dedup: SignalDedupGate,
        m15_reversal_block: M15ReversalBlockGate | None = None,
        analyze_symbol: AnalyzeSymbolFn,
        candle_limit: int,
        poll_interval_seconds: float = 60.0,
        scan_interval_seconds: float = 900.0,
        publish_signal: Callable[..., ActiveTrade] | None = None,
        analyze_scalp: AnalyzeScalpFn | None = None,
        publish_scalp_signal: PublishScalpFn | None = None,
        analyze_premium_scalp: AnalyzeScalpFn | None = None,
        publish_premium_scalp_signal: PublishScalpFn | None = None,
        analyze_turtle_soup_scalp: AnalyzeScalpFn | None = None,
        publish_turtle_soup_scalp_signal: PublishScalpFn | None = None,
        scalp_monitor: TradeMonitor | None = None,
        scalp_dedup: SignalDedupGate | None = None,
        premium_dedup: SignalDedupGate | None = None,
        turtle_dedup: SignalDedupGate | None = None,
        scalp_timeframe: str = SCALP_TIMEFRAME,
        premium_timeframe: str = SWEEP_FVG_TIMEFRAME,
        turtle_timeframe: str = TURTLE_SOUP_TIMEFRAME,
        scalp_scan_interval_seconds: float = DEFAULT_SCALP_SCAN_INTERVAL_SECONDS,
        premium_scan_interval_seconds: float = DEFAULT_PREMIUM_SCAN_INTERVAL_SECONDS,
        turtle_scan_interval_seconds: float = DEFAULT_TURTLE_SCAN_INTERVAL_SECONDS,
        loop_sleep_seconds: float = 1.0,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.logger = logger
        self.provider = provider
        self.signal_filter = signal_filter
        self.signal_generator = signal_generator
        self.monitor = monitor
        self.dedup = dedup
        self.m15_reversal_block = m15_reversal_block
        self.analyze_symbol = analyze_symbol
        self.candle_limit = candle_limit
        self.poll_interval_seconds = poll_interval_seconds
        self.scan_interval_seconds = scan_interval_seconds
        self.publish_signal_fn = publish_signal
        self.analyze_scalp_fn = analyze_scalp
        self.publish_scalp_signal_fn = publish_scalp_signal
        self.analyze_premium_scalp_fn = analyze_premium_scalp
        self.publish_premium_scalp_signal_fn = publish_premium_scalp_signal
        self.analyze_turtle_soup_scalp_fn = analyze_turtle_soup_scalp
        self.publish_turtle_soup_scalp_signal_fn = publish_turtle_soup_scalp_signal
        self.scalp_monitor = scalp_monitor
        self.scalp_dedup = scalp_dedup if scalp_dedup is not None else dedup
        self.premium_dedup = premium_dedup
        self.turtle_dedup = turtle_dedup
        self.scalp_timeframe = scalp_timeframe
        self.premium_timeframe = premium_timeframe
        self.turtle_timeframe = turtle_timeframe
        self.scalp_scan_interval_seconds = scalp_scan_interval_seconds
        self.premium_scan_interval_seconds = premium_scan_interval_seconds
        self.turtle_scan_interval_seconds = turtle_scan_interval_seconds
        self.loop_sleep_seconds = loop_sleep_seconds

    def run_forever(self) -> None:
        self.logger.info(
            "Continuous mode started | symbols=%s | poll=%.0fs | scan=%.0fs",
            ", ".join(self.symbols),
            self.poll_interval_seconds,
            self.scan_interval_seconds,
        )
        safe_print()
        safe_print("=== CONTINUOUS MODE ===")
        safe_print(f"Symbols: {', '.join(self.symbols)}")
        safe_print(f"Poll interval: {self.poll_interval_seconds:.0f}s")
        safe_print(f"Scan interval: {self.scan_interval_seconds:.0f}s")
        safe_print(f"Scalp scan interval: {self.scalp_scan_interval_seconds:.0f}s")
        if self.analyze_premium_scalp_fn is not None:
            safe_print(f"VIP premium scan interval: {self.premium_scan_interval_seconds:.0f}s")
        if self.analyze_turtle_soup_scalp_fn is not None:
            safe_print(f"VIP2 Turtle Soup scan interval: {self.turtle_scan_interval_seconds:.0f}s")
        safe_print("Press Ctrl+C to stop.")
        safe_print()

        self._scan_all_symbols()
        if self.analyze_scalp_fn is not None:
            self._scan_scalp_all_symbols()
        if self.analyze_premium_scalp_fn is not None:
            self._scan_premium_scalp_all_symbols()
        if self.analyze_turtle_soup_scalp_fn is not None:
            self._scan_turtle_soup_scalp_all_symbols()
        last_poll = 0.0
        last_scan = time.monotonic()
        last_scalp_scan = time.monotonic()
        last_premium_scan = time.monotonic()
        last_turtle_scan = time.monotonic()

        try:
            while True:
                now = time.monotonic()

                if now - last_poll >= self.poll_interval_seconds:
                    self._tick_monitors()
                    last_poll = now

                if now - last_scan >= self.scan_interval_seconds:
                    self._scan_all_symbols()
                    last_scan = now

                if (
                    self.analyze_scalp_fn is not None
                    and now - last_scalp_scan >= self.scalp_scan_interval_seconds
                ):
                    self._scan_scalp_all_symbols()
                    last_scalp_scan = now

                if (
                    self.analyze_premium_scalp_fn is not None
                    and now - last_premium_scan >= self.premium_scan_interval_seconds
                ):
                    self._scan_premium_scalp_all_symbols()
                    last_premium_scan = now

                if (
                    self.analyze_turtle_soup_scalp_fn is not None
                    and now - last_turtle_scan >= self.turtle_scan_interval_seconds
                ):
                    self._scan_turtle_soup_scalp_all_symbols()
                    last_turtle_scan = now

                time.sleep(self.loop_sleep_seconds)
        except KeyboardInterrupt:
            self.logger.info("Continuous mode stopped by user")
            safe_print()
            safe_print("Continuous mode stopped.")

    def run_once(self, *, max_polls: int | None = None) -> None:
        self.logger.info("Single-pass mode | symbols=%s", ", ".join(self.symbols))
        self._scan_all_symbols()

        open_trades = self._open_trades()
        if not open_trades:
            self.logger.info("No active trades to monitor")
            return

        safe_print()
        safe_print("=== TRADE MONITOR STARTED ===")
        for trade in open_trades:
            safe_print(f"Monitoring {trade.symbol} {trade.direction.value.upper()} trade...")

        polls = 0
        while self._open_trades():
            self._tick_monitors()
            polls += 1
            if max_polls is not None and polls >= max_polls:
                safe_print()
                safe_print("Monitoring stopped (max polls reached). Open trades remain active.")
                break
            time.sleep(self.poll_interval_seconds)

        self.logger.info("Single-pass monitoring completed")

    def _open_symbols(self) -> set[str]:
        return {
            resolve_symbol(trade.symbol).display
            for trade in self._open_trades()
        }

    def _open_trades(self) -> list[ActiveTrade]:
        return [trade for trade in self.monitor.active_trades if not trade.closed]

    def _scalp_open_symbols(self, *, timeframe: str | None = None) -> set[str]:
        monitor = self.scalp_monitor if self.scalp_monitor is not None else self.monitor
        trades = [trade for trade in monitor.active_trades if not trade.closed]
        if timeframe is not None:
            trades = [trade for trade in trades if trade.timeframe == timeframe]
        return {
            resolve_symbol(trade.symbol).display
            for trade in trades
        }

    def _tick_monitors(self) -> None:
        closed_trades = self.monitor.tick_all()
        for trade in closed_trades:
            self.logger.info(
                "Trade closed | %s | %s | result=%s",
                trade.symbol,
                trade.direction.value,
                trade.result,
            )

        if self.scalp_monitor is not None:
            scalp_closed = self.scalp_monitor.tick_all()
            for trade in scalp_closed:
                self.logger.info(
                    "Scalp trade closed | %s | %s | result=%s",
                    trade.symbol,
                    trade.direction.value,
                    trade.result,
                )

    def _scan_all_symbols(self) -> None:
        open_symbols = self._open_symbols()
        for symbol in self.symbols:
            display_symbol = resolve_symbol(symbol).display
            if display_symbol in open_symbols:
                self.logger.debug(
                    "Skipping scan for %s: open trade already active",
                    display_symbol,
                )
                continue

            signal, results, filter_result, context = self.analyze_symbol(
                symbol,
                provider=self.provider,
                timeframe=self.timeframe,
                candle_limit=self.candle_limit,
                signal_filter=self.signal_filter,
                signal_generator=self.signal_generator,
                logger=self.logger,
            )
            if signal is None or results is None or filter_result is None:
                if filter_result is not None and not filter_result.approved:
                    self.logger.info(
                        "Main scan skipped for %s: %s",
                        display_symbol,
                        filter_result.message,
                    )
                continue

            decision = self.dedup.can_publish(symbol, signal, self._open_symbols())
            if not decision.allowed:
                self.logger.info(
                    "Signal skipped for %s: %s",
                    display_symbol,
                    decision.reason,
                )
                continue

            if self.m15_reversal_block is not None:
                block_decision = self.m15_reversal_block.can_publish(
                    symbol,
                    signal,
                    self.timeframe,
                )
                if not block_decision.allowed:
                    self.logger.info(
                        "Signal skipped for %s: %s",
                        display_symbol,
                        block_decision.reason,
                    )
                    continue

            self._publish_trade(
                symbol,
                signal,
                results=results,
                filter_result=filter_result,
                context=context,
            )
            self.dedup.record_published(symbol, signal)
            self.logger.info("Signal published for %s", display_symbol)

    def _scan_scalp_all_symbols(self) -> None:
        if self.analyze_scalp_fn is None:
            return

        open_symbols = self._scalp_open_symbols(timeframe=self.scalp_timeframe)
        for symbol in self.symbols:
            display_symbol = resolve_symbol(symbol).display
            if not is_scalp_enabled(display_symbol):
                continue
            if display_symbol in open_symbols:
                self.logger.debug(
                    "Skipping scalp scan for %s: open scalp trade already active",
                    display_symbol,
                )
                continue

            signal, context, scalp_result = self.analyze_scalp_fn(
                symbol,
                provider=self.provider,
            )
            if signal is None or scalp_result is None:
                if scalp_result is not None and scalp_result.message:
                    self.logger.info(
                        "Scalp skipped for %s: %s",
                        display_symbol,
                        scalp_result.message,
                    )
                continue

            decision = self.scalp_dedup.can_publish(
                symbol,
                signal,
                self._scalp_open_symbols(timeframe=self.scalp_timeframe),
            )
            if not decision.allowed:
                self.logger.info(
                    "Scalp skipped for %s: %s",
                    display_symbol,
                    decision.reason,
                )
                continue

            if self.m15_reversal_block is not None:
                block_decision = self.m15_reversal_block.can_publish(
                    symbol,
                    signal,
                    self.scalp_timeframe,
                )
                if not block_decision.allowed:
                    self.logger.info(
                        "Scalp skipped for %s: %s",
                        display_symbol,
                        block_decision.reason,
                    )
                    continue

            agents_agreement = format_agents_agreement(
                scalp_result.m5_results or {},
                scalp_result.direction,
            )
            self._publish_scalp_trade(
                symbol,
                signal,
                agents_agreement=agents_agreement,
                context=context,
                m5_results=scalp_result.m5_results,
            )
            self.scalp_dedup.record_published(symbol, signal)
            self.logger.info("Scalp signal published for %s", display_symbol)

    def _publish_scalp_trade(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        agents_agreement: str,
        context: dict | None,
        m5_results: dict[str, AgentResult] | None,
    ) -> ActiveTrade:
        if self.publish_scalp_signal_fn is not None:
            return self.publish_scalp_signal_fn(
                symbol,
                signal,
                agent_results=m5_results,
                agents_agreement=agents_agreement,
                context=context,
                timeframe=self.scalp_timeframe,
            )

        trade = ActiveTrade.from_signal(
            symbol,
            signal,
            agents_agreement=agents_agreement,
            timeframe=self.scalp_timeframe,
            entry_agent_results=m5_results,
        )
        monitor = self.scalp_monitor if self.scalp_monitor is not None else self.monitor
        monitor.register_trade(trade, context=context)
        return trade

    def _scan_premium_scalp_all_symbols(self) -> None:
        if self.analyze_premium_scalp_fn is None or self.premium_dedup is None:
            return

        open_symbols = self._scalp_open_symbols(timeframe=self.premium_timeframe)
        for symbol in self.symbols:
            display_symbol = resolve_symbol(symbol).display
            if not is_sweep_fvg_scalp_enabled(display_symbol):
                continue
            if display_symbol in open_symbols:
                self.logger.debug(
                    "Skipping VIP scalp for %s: open premium trade already active",
                    display_symbol,
                )
                continue

            signal, context, premium_result = self.analyze_premium_scalp_fn(
                symbol,
                provider=self.provider,
            )
            if signal is None or premium_result is None:
                if premium_result is not None and premium_result.message:
                    self.logger.info(
                        "VIP scalp skipped for %s: %s",
                        display_symbol,
                        premium_result.message,
                    )
                continue

            decision = self.premium_dedup.can_publish(
                symbol,
                signal,
                self._scalp_open_symbols(timeframe=self.premium_timeframe),
            )
            if not decision.allowed:
                self.logger.info(
                    "VIP scalp skipped for %s: %s",
                    display_symbol,
                    decision.reason,
                )
                continue

            self._publish_premium_scalp_trade(symbol, signal, context=context)
            self.premium_dedup.record_published(symbol, signal)
            self.logger.info("VIP premium scalp published for %s", display_symbol)

    def _publish_premium_scalp_trade(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        context: dict | None,
    ) -> ActiveTrade:
        if self.publish_premium_scalp_signal_fn is not None:
            return self.publish_premium_scalp_signal_fn(
                symbol,
                signal,
                context=context,
                timeframe=self.premium_timeframe,
            )

        trade = ActiveTrade.from_signal(
            symbol,
            signal,
            agents_agreement="VIP",
            timeframe=self.premium_timeframe,
            entry_agent_results=None,
        )
        monitor = self.scalp_monitor if self.scalp_monitor is not None else self.monitor
        monitor.register_trade(trade, context=context)
        return trade

    def _scan_turtle_soup_scalp_all_symbols(self) -> None:
        if self.analyze_turtle_soup_scalp_fn is None or self.turtle_dedup is None:
            return

        open_symbols = self._scalp_open_symbols(timeframe=self.turtle_timeframe)
        for symbol in self.symbols:
            display_symbol = resolve_symbol(symbol).display
            if not is_turtle_soup_scalp_enabled(display_symbol):
                continue
            if display_symbol in open_symbols:
                self.logger.debug(
                    "Skipping VIP2 Turtle Soup for %s: open trade already active",
                    display_symbol,
                )
                continue

            signal, context, turtle_result = self.analyze_turtle_soup_scalp_fn(
                symbol,
                provider=self.provider,
            )
            if signal is None or turtle_result is None:
                if turtle_result is not None and turtle_result.message:
                    self.logger.info(
                        "VIP2 Turtle Soup skipped for %s: %s",
                        display_symbol,
                        turtle_result.message,
                    )
                continue

            decision = self.turtle_dedup.can_publish(
                symbol,
                signal,
                self._scalp_open_symbols(timeframe=self.turtle_timeframe),
            )
            if not decision.allowed:
                self.logger.info(
                    "VIP2 Turtle Soup skipped for %s: %s",
                    display_symbol,
                    decision.reason,
                )
                continue

            self._publish_turtle_soup_scalp_trade(symbol, signal, context=context)
            self.turtle_dedup.record_published(symbol, signal)
            self.logger.info("VIP2 Turtle Soup published for %s", display_symbol)

    def _publish_turtle_soup_scalp_trade(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        context: dict | None,
    ) -> ActiveTrade:
        if self.publish_turtle_soup_scalp_signal_fn is not None:
            return self.publish_turtle_soup_scalp_signal_fn(
                symbol,
                signal,
                context=context,
                timeframe=self.turtle_timeframe,
            )

        trade = ActiveTrade.from_signal(
            symbol,
            signal,
            agents_agreement="VIP2",
            timeframe=self.turtle_timeframe,
            entry_agent_results=None,
        )
        monitor = self.scalp_monitor if self.scalp_monitor is not None else self.monitor
        monitor.register_trade(trade, context=context)
        return trade

    def _publish_trade(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        results: dict[str, AgentResult],
        filter_result: FilterResult,
        context: dict | None = None,
    ) -> ActiveTrade:
        agents_agreement = format_agents_agreement(
            results,
            filter_result.direction,
            config=self.signal_filter.decision_config,
        )
        if self.publish_signal_fn is not None:
            return self.publish_signal_fn(
                symbol,
                signal,
                timeframe=self.timeframe,
                agent_results=results,
                agents_agreement=agents_agreement,
                news_warning=filter_result.news_warning,
                off_hours_warning=filter_result.off_hours_warning,
                h4_mismatch_warning=filter_result.h4_mismatch_warning,
                context=context,
            )

        trade = ActiveTrade.from_signal(
            symbol,
            signal,
            agents_agreement=agents_agreement,
            timeframe=self.timeframe,
            entry_agent_results=results,
        )
        self.monitor.register_trade(trade, context=context)
        return trade
