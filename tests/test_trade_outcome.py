import unittest

from tracking.trade_outcome import (
    is_full_stop_loss_from_values,
    is_full_stop_loss_record,
)


class FullStopLossClassificationTests(unittest.TestCase):
    def test_full_stop_at_initial_sl(self):
        self.assertTrue(
            is_full_stop_loss_from_values(
                result="stop_loss",
                tp1_hit=False,
                pnl_r=-1.0,
                exit_price=90.0,
                entry=100.0,
                initial_stop_loss=90.0,
                risk=10.0,
            )
        )

    def test_breakeven_at_entry_is_not_full_stop(self):
        self.assertFalse(
            is_full_stop_loss_from_values(
                result="breakeven",
                tp1_hit=False,
                pnl_r=0.0,
                exit_price=100.0,
                entry=100.0,
                initial_stop_loss=90.0,
                risk=10.0,
            )
        )

    def test_trailing_exit_after_tp1_is_not_full_stop(self):
        self.assertFalse(
            is_full_stop_loss_from_values(
                result="stop_loss",
                tp1_hit=True,
                pnl_r=0.75,
                exit_price=100.0,
                entry=100.0,
                initial_stop_loss=90.0,
                risk=10.0,
            )
        )

    def test_record_full_stop(self):
        self.assertTrue(
            is_full_stop_loss_record(
                type(
                    "_R",
                    (),
                    {
                        "result": "stop_loss",
                        "tp1_hit": False,
                        "entry": 100.0,
                        "stop_loss": 90.0,
                    },
                )()
            )
        )


if __name__ == "__main__":
    unittest.main()
