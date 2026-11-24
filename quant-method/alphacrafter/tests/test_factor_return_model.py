import unittest

import numpy as np
import pandas as pd

from alphax.factor_dsl import Candidate
from cn_portfolio import CNConfig, stamp_duty
from factor_return_model import (
    ENTRY, ENTRY_OPEN, EXIT, TARGET, TOP_K, WeeklyEvaluation, performance, review_scores, samples,
)
from run_factor_return_review import coverage, extra_features


class FactorReturnModelTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2009-12-29", periods=25)
        self.candidate = Candidate("x", "feature", ("x",), "fixture")
        dates = self.calendar[:-6]
        self.panel = pd.DataFrame([
            {"date": day, "ticker": str(ticker), "x": ticker, ENTRY: self.calendar[i + 1],
             EXIT: self.calendar[i + 6], ENTRY_OPEN: 100., TARGET: .10 if ticker else 1.}
            for i, day in enumerate(dates) for ticker in range(TOP_K + 1)
        ])

    def test_signal_selects_top_k_and_charges_fees(self):
        first_fill = self.calendar[self.calendar.searchsorted("2010-01-01")]
        signal = self.calendar[self.calendar.get_loc(first_fill) - 1]
        right = self.panel.date.eq(signal)
        self.panel.loc[right & self.panel.x.eq(TOP_K), [ENTRY_OPEN, TARGET]] = np.nan
        runner = WeeklyEvaluation(self.panel, self.calendar, "2010-01-20")
        weeks = runner.weekly(self.candidate)
        config = CNConfig(TOP_K)
        side = config.commission + config.slippage
        exit_day = weeks.loc[first_fill, "exit"]
        net = 1.10 * (1 - side - stamp_duty(exit_day)) / (1 + side) - 1
        self.assertAlmostEqual(weeks.loc[first_fill, "net"], net * (TOP_K - 1) / TOP_K)
        self.assertAlmostEqual(weeks.loc[first_fill, "filled_share"], (TOP_K - 1) / TOP_K)
        self.panel.loc[right & self.panel.x.eq(TOP_K - 1), TARGET] = np.nan
        unknown = WeeklyEvaluation(self.panel, self.calendar, "2010-01-20").weekly(self.candidate)
        self.assertTrue(pd.isna(unknown.loc[first_fill, "net"]))

    def test_future_outcomes_do_not_change_scoring_features(self):
        cutoff = pd.Timestamp("2010-01-15")
        before = WeeklyEvaluation(self.panel, self.calendar, cutoff).predict_features(self.candidate, cutoff)
        self.assertIsNotNone(before)
        self.panel.loc[self.panel[EXIT].ge(cutoff), TARGET] = -0.99
        after = WeeklyEvaluation(self.panel, self.calendar, cutoff).predict_features(self.candidate, cutoff)
        self.assertEqual(before, after)

    def test_coverage_separates_unpriced_exits_from_cash_entries(self):
        first_fill = self.calendar[self.calendar.searchsorted("2010-01-01")]
        signal_pos = self.calendar.get_loc(first_fill) - 1
        entry_holes = self.panel.date.eq(self.calendar[signal_pos]) & self.panel.x.eq(TOP_K)
        exit_holes = self.panel.date.eq(self.calendar[signal_pos + 5]) & self.panel.x.eq(TOP_K)
        score_holes = self.panel.date.eq(self.calendar[signal_pos + 10]) & self.panel.x.ge(2)
        self.panel.loc[entry_holes, [ENTRY_OPEN, TARGET]] = np.nan
        self.panel.loc[exit_holes, TARGET] = np.nan
        self.panel.loc[score_holes, "x"] = np.nan
        weeks = WeeklyEvaluation(self.panel, self.calendar, "2010-02-01").weekly(self.candidate, audit=True)
        result = coverage(weeks.iloc[:3])
        self.assertEqual(result["observed_weeks"], 1)
        self.assertEqual(result["weeks_with_missing_entry"], 1)
        self.assertEqual(result["missing_exit_weeks"], 1)
        self.assertEqual(result["too_few_finite_input_weeks"], 1)

    def test_extended_features_ignore_market_returns_after_cutoff(self):
        origin = pd.Timestamp("2020-01-01")
        dates = pd.date_range("2018-01-01", "2020-02-01", freq="W-MON")
        weeks = pd.DataFrame({"exit": dates + pd.Timedelta(days=7), "net": .01}, index=dates)
        market = pd.Series(.005, index=dates)
        before = extra_features(weeks, market, origin)
        future = weeks.exit.ge(origin)
        weeks.loc[future, "net"] = -.5
        market.loc[future] = .5
        self.assertEqual(before, extra_features(weeks, market, origin))

    def test_training_targets_stop_before_review_and_missing_top_scores_stay_visible(self):
        weeks = pd.date_range("2010-01-04", "2025-12-29", freq="W-MON")
        outcomes = pd.DataFrame({"exit": weeks + pd.Timedelta(days=7), "net": .001, "filled_share": 1.},
                                index=weeks)

        class Evaluation:
            window = staticmethod(WeeklyEvaluation.window)

            def weekly(self, candidate):
                return outcomes

        training, review, four_year = samples(Evaluation(), [self.candidate])
        self.assertLessEqual(training.target_end_exclusive.max(), "2022-01-01")
        self.assertEqual(set(review.origin), {"2022-01-01", "2024-01-01"})
        self.assertEqual(len(four_year), 1)
        self.assertAlmostEqual(training.target.iloc[0], (1.001 ** (252 / 5)) - 1)
        self.assertGreater(performance(outcomes)["cumulative_net"], training.target.iloc[0])
        review["prediction"] = .5
        review.loc[review.origin.eq("2024-01-01"), "target"] = np.nan
        summary = review_scores(review, 1)
        self.assertEqual(summary["2024-01-01"]["scored"], 1)
        self.assertEqual(summary["2024-01-01"]["prediction"]["measured_top_k"], 0)
