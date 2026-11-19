"""Notional-only execution ledger for the explicitly exploratory local demo."""
import unittest

import pandas as pd

from run_agent_execution_demo import simulated_orders
from trader_lite import TraderConfig, simulate


class SimulatedOrdersTests(unittest.TestCase):
    def panel(self):
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-03"] * 2 + ["2023-01-04"] * 2),
            "ticker": ["A", "B", "A", "B"],
            "prediction": [2., 1., 1., 2.],
            "target_open_close": [.1, -.1, .02, -.02],
            "execution_date": pd.to_datetime(["2023-01-04"] * 2 + ["2023-01-05"] * 2),
        })

    def test_orders_reconcile_to_trader_and_follow_next_session(self):
        frame = self.panel()
        report, positions = simulate(frame, TraderConfig(.1, 1., 1, 0),
                                     "2023-01-03", "2023-01-05", .0001)
        orders = simulated_orders(frame, report, positions, .0001, initial_cash=1000.)
        self.assertEqual(len(orders), 4)
        self.assertEqual(orders["leg"].tolist(), ["open", "close", "open", "close"])
        self.assertEqual(orders["ticker"].tolist(), ["A", "A", "B", "B"])
        self.assertEqual(str(orders.iloc[0]["execution_date"].date()), "2023-01-04")
        self.assertAlmostEqual(orders.iloc[0]["notional"], 100.)
        self.assertAlmostEqual(orders.iloc[1]["notional"], 110.)
        self.assertAlmostEqual(orders.iloc[2]["notional"], 1000. * (1 + report.iloc[0]["net_return"]) * .1)

    def test_missing_return_and_mismatched_ledger_fail_closed(self):
        frame = self.panel()
        report, positions = simulate(frame, TraderConfig(.1, 1., 1, 0),
                                     "2023-01-03", "2023-01-05", .0001)
        changed = frame.copy()
        changed.loc[0, "target_open_close"] = float("nan")
        with self.assertRaisesRegex(ValueError, "execution return"):
            simulated_orders(changed, report, positions, .0001)
        bad = report.copy()
        bad.loc[bad.index[0], "transaction_cost"] = 0.
        with self.assertRaisesRegex(ValueError, "reconcile"):
            simulated_orders(frame, bad, positions, .0001)


if __name__ == "__main__":
    unittest.main()
