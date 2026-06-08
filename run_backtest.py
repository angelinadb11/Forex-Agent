import argparse

from backtest import BacktestConfig, BacktestEngine
from backtest.engine import DEFAULT_BTC_TIMEFRAMES
from config import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES, load_settings, resolve_symbol, resolve_timeframe
from tracking.console import configure_console_encoding


def parse_args(settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategy backtest on historical data")
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        choices=[*SUPPORTED_SYMBOLS, "US30", "XAUUSDT"],
        help="Symbol to backtest (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        choices=list(SUPPORTED_TIMEFRAMES),
        help="Single timeframe to test (default: run 1m, 5m, and 15m for BTCUSDT)",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=1500,
        help="Number of historical candles to load per timeframe (default: 1500)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup candles before generating signals (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    settings = load_settings()
    args = parse_args(settings)

    symbol_def = resolve_symbol(args.symbol)
    engine = BacktestEngine(
        BacktestConfig(
            symbol=symbol_def.display,
            timeframe=args.timeframe or "15m",
            total_candles=args.candles,
            warmup_candles=args.warmup,
        )
    )

    if args.timeframe is None and symbol_def.display == "BTCUSDT":
        print(
            f"Running BTCUSDT backtest on {', '.join(DEFAULT_BTC_TIMEFRAMES)} "
            f"({args.candles} candles each)..."
        )
        print()
        payload = engine.run_btcusdt_suite(
            timeframes=DEFAULT_BTC_TIMEFRAMES,
            total_candles=args.candles,
        )
    elif args.timeframe is None:
        timeframes = DEFAULT_BTC_TIMEFRAMES
        print(
            f"Running {symbol_def.display} backtest on {', '.join(timeframes)} "
            f"({args.candles} candles each)..."
        )
        print()
        payload = engine.run_timeframes(timeframes)
    else:
        timeframe = resolve_timeframe(args.timeframe)
        print(
            f"Running backtest on {symbol_def.display} "
            f"({symbol_def.data_symbol}) {timeframe} ({args.candles} candles)..."
        )
        print()
        payload = engine.run()

    engine.print_report(payload)


if __name__ == "__main__":
    main()
