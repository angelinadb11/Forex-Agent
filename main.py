import argparse
import logging

from agents.base import AgentResult
from agents.zone_helpers import ZoneCatalog
from config import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES, load_settings, resolve_symbol, resolve_symbols, resolve_timeframe
from config.logging_config import setup_logging
from config.settings import Settings
from data import MarketDataProvider
from news import build_news_gate
from news.calendar_provider import FOREX_FACTORY_CALENDAR_URL
from runtime import BotRuntime, SignalDedupGate
from signal_generator import SignalGenerator, TradeSignal
from strategy import (
    build_signal_reason,
    compute_final_decision,
    format_agents_agreement,
    run_agents,
    SignalFilter,
)
from runtime.m15_reversal_block import M15ReversalBlockGate
from strategy.liquidity_scalp import (
    LIQUIDITY_SCALP_TIMEFRAME,
    analyze_liquidity_scalp_symbol,
    build_liquidity_scalp_gate,
)
from strategy.sweep_fvg_scalp import (
    SWEEP_FVG_TIMEFRAME,
    analyze_sweep_fvg_scalp_symbol,
    build_premium_scalp_gate,
)
from strategy.turtle_soup_scalp import (
    TURTLE_SOUP_TIMEFRAME,
    analyze_turtle_soup_scalp_symbol,
    build_turtle_soup_gate,
)
from strategy.signal_filter import (
    MAIN_CHANNEL_FILTER_PROFILE,
    MIN_CONFIDENCE,
    MIN_CONFIDENCE_PCT,
    profile_symbols,
)
from config.settings import PROJECT_ROOT
from tracking import TradeMonitor, print_trade_signal
from tracking.active_trades_store import ActiveTradesStore
from tracking.console import safe_print, configure_console_encoding
from tracking.trade_history import TradeHistoryStore
from telegram import TelegramBot, TelegramTradeManager


def parse_args(settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SMC AI Trading Agent")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis pass and monitor any open trades, then exit",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        choices=[*SUPPORTED_SYMBOLS, *["US30", "XAUUSDT"]],
        help=(
            "Analyze a single symbol instead of the configured symbol list. "
            f"Supported: {', '.join(SUPPORTED_SYMBOLS)}"
        ),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        choices=[*SUPPORTED_SYMBOLS, "US30", "XAUUSDT"],
        help="Override configured symbol list for this run",
    )
    parser.add_argument(
        "--timeframe",
        default=settings.timeframe,
        choices=list(SUPPORTED_TIMEFRAMES),
        help=f"Candle timeframe (default from config: {settings.timeframe})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=settings.poll_interval_seconds,
        help="Seconds between monitor ticks (default from config: 60)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=settings.scan_interval_seconds,
        help="Seconds between signal scans in continuous mode (default from config: 900)",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=None,
        help="Optional monitor poll limit for --once mode",
    )
    return parser.parse_args()


def resolve_runtime_symbols(settings: Settings, args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols:
        return resolve_symbols(args.symbols)
    if args.symbol:
        return (resolve_symbol(args.symbol).display,)
    return settings.symbols


def print_agent_result(label: str, result: AgentResult) -> None:
    print(f"{label}: {result.direction.value.upper()} | confidence={result.confidence:.2f}")
    print(f"  Reason: {result.reason}")


def print_filter_result(symbol: str, filter_result) -> None:
    if filter_result.approved:
        print(f"Signal filter ({symbol}): {filter_result.message}")
        print(
            f"Confidence check: PASSED "
            f"({filter_result.confidence:.2f} >= {MIN_CONFIDENCE:.2f} / {MIN_CONFIDENCE_PCT}%)"
        )
        return

    print(f"=== NO TRADE | {symbol} ===")
    print(f"Reason: {filter_result.message}")
    print(
        f"Final confidence: {filter_result.confidence:.2f} "
        f"(minimum: {MIN_CONFIDENCE:.2f} / {MIN_CONFIDENCE_PCT}%)"
    )


def analyze_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    timeframe: str,
    candle_limit: int,
    signal_filter: SignalFilter,
    signal_generator: SignalGenerator,
    logger: logging.Logger,
) -> tuple[TradeSignal | None, dict[str, AgentResult] | None, object | None, dict | None]:
    symbol_def = resolve_symbol(symbol)
    display_symbol = symbol_def.display
    data_source = provider.data_source(display_symbol)

    logger.info(
        "Loading %s (%s) %s candles from %s",
        display_symbol,
        symbol_def.data_symbol,
        timeframe,
        data_source,
    )

    context = provider.to_context(display_symbol, timeframe, limit=candle_limit)
    candles = context["candles"]
    context["zone_catalog"] = ZoneCatalog.from_candles(candles, display_symbol)
    context["bar_index"] = len(candles) - 1
    logger.info("Loaded %s candles for %s", len(candles), display_symbol)

    results = run_agents(context)
    final_direction, final_confidence, long_score, short_score = compute_final_decision(
        results,
        signal_filter.decision_config,
    )

    safe_print()
    safe_print(f"Symbol: {display_symbol}")
    safe_print(f"Timeframe: {timeframe}")
    safe_print(f"Candles loaded: {len(candles)}")
    safe_print()

    print_agent_result("SMC", results["smc"])
    safe_print()
    print_agent_result("Liquidity", results["liquidity"])
    safe_print()
    print_agent_result("FVG", results["fvg"])
    safe_print()
    print_agent_result("Order Block", results["order_block"])
    safe_print()
    print_agent_result("Trend Filter (H1)", results["trend_filter"])
    if "h4_trend_filter" in results:
        safe_print()
        print_agent_result("Trend Filter (H4)", results["h4_trend_filter"])
    safe_print()
    print_agent_result("RSI", results["rsi"])
    safe_print()
    print_agent_result("Session", results["session"])
    safe_print()

    safe_print(f"LONG score: {long_score:.2f}")
    safe_print(f"SHORT score: {short_score:.2f}")
    safe_print()
    safe_print(f"Final decision: {final_direction.value.upper()}")
    safe_print(f"Final confidence: {final_confidence:.2f}")
    safe_print()

    filter_result = signal_filter.evaluate(
        results,
        final_direction,
        final_confidence,
        symbol=display_symbol,
        timestamp=context.get("timestamp"),
        context=context,
    )
    print_filter_result(display_symbol, filter_result)
    safe_print()

    for name, result in results.items():
        logger.info(
            "%s | %s | %s | confidence=%.2f | %s",
            display_symbol,
            name,
            result.direction.value,
            result.confidence,
            result.reason,
        )

    logger.info(
        "%s | FINAL | %s | confidence=%.2f | long=%.2f short=%.2f",
        display_symbol,
        final_direction.value,
        final_confidence,
        long_score,
        short_score,
    )

    if not filter_result.approved:
        return None, results, filter_result, context

    signal_reason = build_signal_reason(results, filter_result.direction)
    generation = signal_generator.generate(
        context,
        filter_result.direction,
        filter_result.confidence,
        signal_reason,
    )
    if generation.signal is None:
        safe_print("=== TRADE SIGNAL ===")
        safe_print(f"Signal not generated for {display_symbol}: {generation.rejection_reason}")
        logger.warning(
            "Signal generation failed for %s: %s",
            display_symbol,
            generation.rejection_reason,
        )
        return None, results, filter_result, context

    signal = generation.signal
    print_trade_signal(display_symbol, signal)
    logger.info(
        "%s | SIGNAL | %s | entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f tp3=%.2f | %s",
        display_symbol,
        signal.direction.value,
        signal.entry,
        signal.stop_loss,
        signal.tp1,
        signal.tp2,
        signal.tp3,
        signal.reason,
    )
    return signal, results, filter_result, context


def build_bot_runtime(
    *,
    settings: Settings,
    symbols: tuple[str, ...],
    timeframe: str,
    logger: logging.Logger,
    poll_interval: float,
    scan_interval: float,
) -> BotRuntime:
    provider = MarketDataProvider()
    signal_filter = SignalFilter.from_profile(
        MAIN_CHANNEL_FILTER_PROFILE,
        london_ny_session_symbols=settings.london_ny_session_symbols,
        session_confidence_symbols=settings.session_confidence_symbols,
        news_gate=build_news_gate(
            enabled=settings.news_enabled,
            buffer_minutes=settings.news_buffer_minutes,
            finnhub_api_key=settings.finnhub_api_key,
            calendar_url=settings.news_calendar_url or FOREX_FACTORY_CALENDAR_URL,
        ),
    )
    logger.info(
        "Main channel filter: %s",
        MAIN_CHANNEL_FILTER_PROFILE.description,
    )
    signal_generator = SignalGenerator()
    telegram_bot = TelegramBot.from_env()
    context_fetcher = lambda symbol, tf: provider.to_context(
        symbol,
        tf,
        limit=settings.candle_limit,
    )
    m15_reversal_block = M15ReversalBlockGate(context_fetcher=context_fetcher)

    trade_manager = (
        TelegramTradeManager(
            price_fetcher=lambda symbol: provider.get_current_price(symbol),
            candle_fetcher=lambda symbol, tf: provider.get_market_data(symbol, tf, limit=1)[-1],
            telegram_bot=telegram_bot,
            poll_interval=poll_interval,
            context_fetcher=context_fetcher,
            m15_reversal_block=m15_reversal_block,
        )
        if telegram_bot is not None
        else None
    )

    if trade_manager is not None:
        monitor = trade_manager.monitor
    else:
        monitor = TradeMonitor(
            price_fetcher=lambda symbol: provider.get_current_price(symbol),
            candle_fetcher=lambda symbol, tf: provider.get_market_data(symbol, tf, limit=1)[-1],
            telegram_bot=None,
            context_fetcher=context_fetcher,
            m15_reversal_block=m15_reversal_block,
        )

    dedup = SignalDedupGate(
        duplicate_entry_tolerance_pct=settings.duplicate_entry_tolerance_pct,
        signal_cooldown_minutes=settings.signal_cooldown_minutes,
    )
    dedup.seed_from_active_trades(monitor.active_trades)
    m15_reversal_block.seed_from_active_trades(monitor.active_trades)

    publish_signal = trade_manager.publish_signal if trade_manager is not None else None

    # Liquidity scalp stream: separate Telegram bot/channel and separate stores.
    scalp_telegram_bot = TelegramBot.from_scalp_env()

    def resolve_scalp_market_timeframe(timeframe: str) -> str:
        if timeframe.startswith("5m"):
            return "5m"
        if timeframe.startswith("1m"):
            return "1m"
        return timeframe

    scalp_trade_manager = None
    if scalp_telegram_bot is not None:
        scalp_trade_manager = TelegramTradeManager(
            price_fetcher=lambda symbol: provider.get_current_price(symbol),
            candle_fetcher=lambda symbol, tf: provider.get_market_data(
                symbol,
                resolve_scalp_market_timeframe(tf),
                limit=1,
            )[-1],
            telegram_bot=scalp_telegram_bot,
            poll_interval=poll_interval,
            context_fetcher=context_fetcher,
            history_store=TradeHistoryStore(PROJECT_ROOT / "scalp_trade_history.json"),
            active_trades_store=ActiveTradesStore(PROJECT_ROOT / "scalp_active_trades.json"),
        )
        logger.info("Liquidity scalp stream enabled (separate Telegram channel)")
    else:
        logger.info(
            "Liquidity scalp stream disabled: TELEGRAM_SCALP_BOT_TOKEN / "
            "TELEGRAM_SCALP_CHAT_ID not configured"
        )

    analyze_scalp = None
    publish_scalp_signal = None
    analyze_premium_scalp = None
    publish_premium_scalp_signal = None
    analyze_turtle_soup_scalp = None
    publish_turtle_soup_scalp_signal = None
    scalp_monitor = None
    scalp_dedup = None
    premium_dedup = None
    turtle_dedup = None
    if scalp_trade_manager is not None:
        liquidity_scalp_gate = build_liquidity_scalp_gate()
        premium_scalp_gate = build_premium_scalp_gate()
        turtle_soup_gate = build_turtle_soup_gate()

        def analyze_scalp(symbol: str, *, provider):
            return analyze_liquidity_scalp_symbol(
                symbol,
                provider=provider,
                publish_gate=liquidity_scalp_gate,
            )

        def analyze_premium_scalp(symbol: str, *, provider):
            return analyze_sweep_fvg_scalp_symbol(
                symbol,
                provider=provider,
                publish_gate=premium_scalp_gate,
            )

        def analyze_turtle_soup_scalp(symbol: str, *, provider):
            return analyze_turtle_soup_scalp_symbol(
                symbol,
                provider=provider,
                publish_gate=turtle_soup_gate,
            )

        publish_scalp_signal = scalp_trade_manager.publish_scalp_signal
        publish_premium_scalp_signal = scalp_trade_manager.publish_premium_scalp_signal
        publish_turtle_soup_scalp_signal = scalp_trade_manager.publish_turtle_soup_scalp_signal
        scalp_monitor = scalp_trade_manager.monitor
        scalp_dedup = SignalDedupGate(
            duplicate_entry_tolerance_pct=settings.duplicate_entry_tolerance_pct,
            signal_cooldown_minutes=settings.signal_cooldown_minutes,
        )
        scalp_dedup.seed_from_active_trades(
            [t for t in scalp_monitor.active_trades if t.timeframe == LIQUIDITY_SCALP_TIMEFRAME]
        )
        premium_dedup = SignalDedupGate(
            duplicate_entry_tolerance_pct=settings.duplicate_entry_tolerance_pct,
            signal_cooldown_minutes=settings.signal_cooldown_minutes,
        )
        premium_dedup.seed_from_active_trades(
            [t for t in scalp_monitor.active_trades if t.timeframe == SWEEP_FVG_TIMEFRAME]
        )
        turtle_dedup = SignalDedupGate(
            duplicate_entry_tolerance_pct=settings.duplicate_entry_tolerance_pct,
            signal_cooldown_minutes=settings.signal_cooldown_minutes,
        )
        turtle_dedup.seed_from_active_trades(
            [t for t in scalp_monitor.active_trades if t.timeframe == TURTLE_SOUP_TIMEFRAME]
        )
        logger.info(
            "VIP premium Sweep+FVG stream enabled on %s (same scalp channel)",
            SWEEP_FVG_TIMEFRAME,
        )
        logger.info(
            "VIP2 Turtle Soup stream enabled on %s (same scalp channel)",
            TURTLE_SOUP_TIMEFRAME,
        )

    return BotRuntime(
        symbols=symbols,
        timeframe=timeframe,
        logger=logger,
        provider=provider,
        signal_filter=signal_filter,
        signal_generator=signal_generator,
        monitor=monitor,
        dedup=dedup,
        m15_reversal_block=m15_reversal_block,
        analyze_symbol=analyze_symbol,
        candle_limit=settings.candle_limit,
        poll_interval_seconds=poll_interval,
        scan_interval_seconds=scan_interval,
        publish_signal=publish_signal,
        analyze_scalp=analyze_scalp,
        publish_scalp_signal=publish_scalp_signal,
        analyze_premium_scalp=analyze_premium_scalp,
        publish_premium_scalp_signal=publish_premium_scalp_signal,
        analyze_turtle_soup_scalp=analyze_turtle_soup_scalp,
        publish_turtle_soup_scalp_signal=publish_turtle_soup_scalp_signal,
        scalp_monitor=scalp_monitor,
        scalp_dedup=scalp_dedup,
        premium_dedup=premium_dedup,
        turtle_dedup=turtle_dedup,
        scalp_timeframe=LIQUIDITY_SCALP_TIMEFRAME,
        premium_timeframe=SWEEP_FVG_TIMEFRAME,
        turtle_timeframe=TURTLE_SOUP_TIMEFRAME,
    )


def main() -> None:
    configure_console_encoding()
    settings = load_settings()
    args = parse_args(settings)

    symbols = resolve_runtime_symbols(settings, args)
    enabled_symbols = profile_symbols(MAIN_CHANNEL_FILTER_PROFILE, symbols)
    timeframe = resolve_timeframe(args.timeframe)

    logger = setup_logging(settings)
    if enabled_symbols != symbols:
        disabled = set(symbols) - set(enabled_symbols)
        logger.info(
            "Main channel profile: disabled symbols skipped: %s",
            ", ".join(sorted(disabled)),
        )
    symbols = enabled_symbols
    logger.info("Analyzing symbols: %s", ", ".join(symbols))

    runtime = build_bot_runtime(
        settings=settings,
        symbols=symbols,
        timeframe=timeframe,
        logger=logger,
        poll_interval=args.poll_interval,
        scan_interval=args.scan_interval,
    )

    if args.once:
        runtime.run_once(max_polls=args.max_polls)
        return

    runtime.run_forever()


if __name__ == "__main__":
    main()
