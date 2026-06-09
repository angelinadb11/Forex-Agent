import time

from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import WARMUP, LocalDataBacktestEngine
from backtest.engine import BacktestConfig

m15 = load_candles(XAUUSD_M15_30D_FILE, "15m")
h1 = load_candles(XAUUSD_M15_30D_FILE, "1h")

t0 = time.perf_counter()
engine = LocalDataBacktestEngine(
    BacktestConfig(
        symbol="XAUUSD",
        timeframe="15m",
        total_candles=len(m15),
        warmup_candles=WARMUP,
    ),
    m15_candles=m15,
    h1_candles=h1,
)
print(f"init: {time.perf_counter() - t0:.2f}s", flush=True)

t1 = time.perf_counter()
setups, stats = engine._scan_candles(m15)
print(
    f"scan: {time.perf_counter() - t1:.2f}s, setups={len(setups)}, "
    f"neutral={stats.neutral_decisions}, blocked={stats.other_filter_blocked}",
    flush=True,
)

legacy, partial = engine.run_comparison(data_file=str(XAUUSD_M15_30D_FILE))
print(f"comparison total: {time.perf_counter() - t0:.2f}s", flush=True)
print(f"legacy R: {legacy.total_r:+.2f}, partial R: {partial.total_r:+.2f}", flush=True)
