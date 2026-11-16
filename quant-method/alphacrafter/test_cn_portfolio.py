import unittest

import numpy as np
import pandas as pd

from cn_portfolio import CNConfig, simulate_cn, stamp_duty


class CNPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2022-01-03", periods=5)
        self.quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100., "close": 100.}
                                    for d in self.calendar for t in ("A", "B", "SH000300")])
        self.signals = pd.DataFrame([{"date": d, "ticker": t, "prediction": score}
                                     for i, d in enumerate(self.calendar)
                                     for t, score in (("A", float(i % 2 == 0)), ("B", float(i % 2 != 0)))])

    def run_sim(self, signals=None, quotes=None, config=None, end=None):
        return simulate_cn(self.signals if signals is None else signals,
                           self.quotes if quotes is None else quotes, self.calendar,
                           config or CNConfig(1, commission=0., slippage=0.), self.calendar[1], end or self.calendar[-1])

    def test_only_previous_session_signals_and_no_same_day_resale(self):
        _, trades, _, audit = self.run_sim()
        first = trades[trades.date == self.calendar[1]]
        self.assertEqual(first.ticker.tolist(), ["A"])
        self.assertEqual(first.signal_date.tolist(), [self.calendar[0]])
        bought = {}
        for row in trades.itertuples():
            if row.side == "buy":
                bought[row.ticker] = row.date
            else:
                self.assertLess(bought[row.ticker], row.date)
        self.assertTrue(audit["liquidation_complete"])

    def test_costs_include_entry_exit_slippage_and_sell_tax(self):
        report, trades, _, _ = self.run_sim(config=CNConfig(1), end=self.calendar[2])
        expected_cost = .8 * (.0003 + .0005) + .8 * (.0003 + .0005 + .001)
        self.assertAlmostEqual(report.nav.iloc[-1], 1 - expected_cost)
        self.assertAlmostEqual(trades.stamp_duty.sum(), .8 * .001)
        self.assertEqual(stamp_duty("2023-08-25"), .001)
        self.assertEqual(stamp_duty("2023-08-28"), .0005)

    def test_untradable_selected_stock_is_not_replaced_with_future_tradable_name(self):
        quotes = self.quotes.copy()
        quotes.loc[(quotes.date == self.calendar[1]) & (quotes.ticker == "A"), "open"] = np.nan
        report, trades, _, _ = self.run_sim(quotes=quotes)
        self.assertTrue(trades[trades.date == self.calendar[1]].empty)
        self.assertEqual(report.loc[self.calendar[1], "cash"], 1.)

    def test_today_close_does_not_change_today_orders(self):
        _, before, _, _ = self.run_sim()
        quotes = self.quotes.copy()
        quotes.loc[(quotes.date == self.calendar[1]) & (quotes.ticker == "A"), "close"] = 120.
        _, after, _, _ = self.run_sim(quotes=quotes)
        pd.testing.assert_frame_equal(before[before.date == self.calendar[1]].reset_index(drop=True),
                                      after[after.date == self.calendar[1]].reset_index(drop=True))

    def test_suspension_marks_are_reported_and_terminal_position_not_erased(self):
        signals = self.signals.copy()
        signals["prediction"] = (signals.ticker == "A").astype(float)
        quotes = self.quotes.copy()
        quotes.loc[(quotes.date >= self.calendar[2]) & (quotes.ticker == "A"), ["open", "close"]] = np.nan
        report, _, positions, audit = self.run_sim(signals=signals, quotes=quotes)
        self.assertFalse(audit["liquidation_complete"])
        self.assertEqual(audit["ending_positions"], 1)
        self.assertGreater(audit["stale_valuation_days"], 0)
        self.assertTrue(positions.stale_mark.any())
        self.assertGreater(report.nav.iloc[-1], 0)

    def test_full_exposure_still_reserves_cash_for_fees(self):
        report, _, _, _ = self.run_sim(config=CNConfig(1, exposure=1.))
        self.assertTrue(report.cash.ge(0).all())

    def test_opening_gap_guard_blocks_buys_and_terminal_sells(self):
        quotes = self.quotes.copy()
        quotes.loc[(quotes.date == self.calendar[1]) & (quotes.ticker == "A"), "open"] = 110.
        _, trades, _, _ = self.run_sim(quotes=quotes)
        self.assertTrue(trades[trades.date == self.calendar[1]].empty)
        quotes = self.quotes.copy()
        quotes.loc[(quotes.date == self.calendar[2]) & (quotes.ticker == "A"), "open"] = 90.
        _, _, _, audit = self.run_sim(quotes=quotes, end=self.calendar[2])
        self.assertFalse(audit["liquidation_complete"])

    def test_no_signals_stays_cash_and_preserves_empty_ledger_schema(self):
        signals = self.signals.copy()
        signals["prediction"] = np.nan
        report, trades, positions, audit = self.run_sim(signals=signals)
        self.assertTrue(report.nav.eq(1).all())
        self.assertTrue(trades.empty)
        self.assertIn("stamp_duty", trades.columns)
        self.assertIn("stale_mark", positions.columns)
        self.assertTrue(audit["liquidation_complete"])


if __name__ == "__main__":
    unittest.main()
