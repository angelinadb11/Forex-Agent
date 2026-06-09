import json
import tempfile
import unittest
from pathlib import Path

from agents.base import AgentResult, Direction
from tracking.trade_history import TradeHistoryStore, TradeRecord


class TradeHistoryAgentResultsTests(unittest.TestCase):
    def test_round_trip_entry_agent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trade_history.json"
            store = TradeHistoryStore(file_path=path)
            results = {
                "liquidity": AgentResult(
                    Direction.LONG,
                    0.45,
                    "XAUUSD 15m Liquidity: sweep",
                ),
                "smc": AgentResult(Direction.LONG, 0.40, "bullish BOS"),
            }
            store.add_trade(
                TradeRecord(
                    symbol="XAUUSD",
                    direction="long",
                    entry=3300.0,
                    stop_loss=3290.0,
                    tp1=3310.0,
                    tp2=3320.0,
                    tp3=3330.0,
                    confidence=0.72,
                    reason="test",
                    open_time="2026-06-07T00:00:00+00:00",
                    close_time="2026-06-07T01:00:00+00:00",
                    result="stop_loss",
                    entry_agent_results=results,
                )
            )

            loaded = store.load()[0]
            self.assertIsNotNone(loaded.entry_agent_results)
            self.assertEqual(loaded.entry_agent_results["liquidity"].direction, Direction.LONG)
            self.assertEqual(loaded.entry_agent_results["smc"].reason, "bullish BOS")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("entry_agent_results", payload[0])
            self.assertIn("liquidity", payload[0]["entry_agent_results"])


if __name__ == "__main__":
    unittest.main()
