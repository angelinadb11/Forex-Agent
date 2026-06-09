import time

from agents.base import Direction
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import WARMUP
from agents.zone_helpers import ZoneCatalog
from backtest.engine import BacktestConfig, candle_timestamp
from run_xauusd_30d_backtest import LocalDataBacktestEngine
from strategy.runner import build_context, compute_final_decision, run_agents, slice_candles_as_of
from strategy.signal_filter import SignalFilter

m15 = load_candles(XAUUSD_M15_30D_FILE, "15m")
h1 = load_candles(XAUUSD_M15_30D_FILE, "1h")
engine = LocalDataBacktestEngine(
    BacktestConfig(symbol="XAUUSD", timeframe="15m", total_candles=len(m15), warmup_candles=WARMUP),
    m15_candles=m15,
    h1_candles=h1,
)
catalog = engine._zone_catalog
sf = SignalFilter()

agents_time = 0.0
filter_time = 0.0
gen_time = 0.0
approved = 0
start = WARMUP
end = start + 200

for index in range(start, end):
    history = m15[: index + 1]
    timestamp = candle_timestamp(m15, index)
    context = build_context("XAUUSD", history, "15m", timestamp, slice_candles_as_of(h1, timestamp))
    context["zone_catalog"] = catalog
    context["bar_index"] = index

    t0 = time.perf_counter()
    agent_results = run_agents(context)
    agents_time += time.perf_counter() - t0

    direction, confidence, _, _ = compute_final_decision(agent_results)
    if direction == Direction.NEUTRAL:
        continue

    t1 = time.perf_counter()
    filter_result = sf.evaluate(agent_results, direction, confidence, symbol="XAUUSD", timestamp=timestamp)
    filter_time += time.perf_counter() - t1
    if not filter_result.approved:
        continue

    t2 = time.perf_counter()
    engine.signal_generator.generate(
        context,
        filter_result.direction,
        filter_result.confidence,
        "test",
    )
    gen_time += time.perf_counter() - t2
    approved += 1

count = end - start
print(f"bars={count} agents={agents_time:.2f}s filter={filter_time:.2f}s gen={gen_time:.2f}s approved={approved}")
