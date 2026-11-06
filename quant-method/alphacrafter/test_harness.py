#!/usr/bin/env python3
"""Small regression tests for Miner-lite temporal and cost boundaries."""

from __future__ import annotations

import unittest

import pandas as pd

from miner_lite import (
    Candidate,
    META_COLUMNS,
    candidate_from_dict,
    derive_target_5d,
    build_accepted_signal,
    evaluate_candidate_score,
    evaluate_score,
    expression_novelty_report,
    library_factor,
    needs_revalidation,
)
from run_miner_walk_forward import (
    build_signal,
    select_factors,
    validation_portfolio_metrics,
)


class MinerHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.Series(
            pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2021-01-04",
                    "2021-01-04",
                    "2021-01-04",
                ]
            )
        )
        self.frame = pd.DataFrame(
            {
                "date": self.dates,
                "ticker": ["A", "B", "C", "A", "B", "C"],
            }
        )
        self.ranked = pd.DataFrame({"feature_x": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8]})
        self.target = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        self.candidates = [
            Candidate("single__feature_x", "feature", ("feature_x",), "test")
        ]

    def test_future_target_changes_do_not_change_train_selection(self) -> None:
        selected = select_factors(
            self.dates,
            self.target,
            self.ranked,
            self.candidates,
            "2020-12-31",
            0.0,
        )
        changed_target = self.target.copy()
        changed_target.iloc[3:] = [100.0, -100.0, 50.0]
        changed = select_factors(
            self.dates,
            changed_target,
            self.ranked,
            self.candidates,
            "2020-12-31",
            0.0,
        )
        self.assertEqual(selected[0]["direction"], changed[0]["direction"])
        self.assertEqual(selected[0]["train_rank_ic"], changed[0]["train_rank_ic"])

    def test_signal_is_truncated_at_evaluation_end(self) -> None:
        selected = [
            {
                "candidate_id": "single__feature_x",
                "op": "feature",
                "features": ["feature_x"],
                "direction": 1,
            }
        ]
        signal = build_signal(self.frame, self.ranked, selected, "2020-12-31")
        self.assertEqual(signal["datetime"].max(), pd.Timestamp("2020-01-02"))

    def test_validation_cost_reduces_net_return(self) -> None:
        low_cost = validation_portfolio_metrics(
            self.dates,
            self.frame["ticker"],
            self.target,
            self.ranked["feature_x"],
            "2020-01-01",
            "2021-12-31",
            2,
            1,
            0.0,
        )
        high_cost = validation_portfolio_metrics(
            self.dates,
            self.frame["ticker"],
            self.target,
            self.ranked["feature_x"],
            "2020-01-01",
            "2021-12-31",
            2,
            1,
            0.01,
        )
        self.assertLess(high_cost["validation_cumulative_return"], low_cost["validation_cumulative_return"])

    def test_future_label_is_metadata(self) -> None:
        self.assertIn("target_open_close", META_COLUMNS)
        self.assertIn("target_5d", META_COLUMNS)

    def test_five_day_target_uses_only_future_sessions(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["A"] * 7,
                "date": pd.date_range("2020-01-01", periods=7),
                "return_1d": [0.01] * 7,
            }
        )
        target = derive_target_5d(frame)
        self.assertAlmostEqual(float(target.iloc[0]), (1.01 ** 5) - 1.0)
        self.assertTrue(pd.isna(target.iloc[-1]))

    def test_safe_time_series_expression_is_evaluated(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "ticker": ["A"] * 5,
                "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        ranked = frame[["feature_x"]].groupby(frame["date"]).rank(pct=True)
        candidate = candidate_from_dict(
            {
                "candidate_id": "ts_delta_feature_x",
                "features": ["feature_x"],
                "expression": {"op": "ts_delta", "feature": "feature_x", "window": 2},
            }
        )
        score = evaluate_candidate_score(candidate, ranked, frame, frame["date"])
        self.assertTrue(pd.isna(score.iloc[0]))
        self.assertEqual(float(score.iloc[2]), 2.0)

    def test_factor_library_revalidation_interval(self) -> None:
        factor = {
            "candidate_id": "single__feature_x",
            "op": "feature",
            "features": ["feature_x"],
            "direction": 1,
        }
        stored = library_factor(factor, None, pd.Timestamp("2020-01-01"), 90)
        self.assertFalse(needs_revalidation(stored, pd.Timestamp("2020-03-30"), 90))
        self.assertTrue(needs_revalidation(stored, pd.Timestamp("2020-04-01"), 90))
        self.assertEqual(stored["validation_count"], 1)
        self.assertEqual(len(stored["validation_history"]), 1)

    def test_expression_novelty_report_is_structural(self) -> None:
        report = expression_novelty_report(
            [
                Candidate("a", "feature", ("feature_a",), ""),
                Candidate("b", "difference", ("feature_a", "feature_b"), ""),
            ]
        )
        self.assertEqual(report["candidate_count"], 2)
        self.assertGreater(report["internal_structural_diversity"], 0.0)

    def test_coverage_and_turnover_are_reported(self) -> None:
        dates = pd.Series(pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3))
        instruments = pd.Series(["A", "B", "C"] * 2)
        score = pd.Series([0.1, 0.5, float("nan"), 0.2, 0.4, 0.8])
        target = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        metric = evaluate_score(
            dates, score, target, pd.Series([True] * len(dates)), instruments
        )
        self.assertAlmostEqual(metric.coverage, 5.0 / 6.0)
        self.assertGreaterEqual(metric.turnover, 0.0)
        self.assertLessEqual(metric.turnover, 1.0)
        self.assertTrue(hasattr(metric, "mean_daily_ic"))
        self.assertTrue(hasattr(metric, "icir"))

    def test_accepted_ast_factor_is_rehydrated_for_signal_building(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3
                ),
                "ticker": ["A", "B", "C"] * 3,
                "feature_x": [1.0, 2.0, 3.0, 2.0, 4.0, 8.0, 3.0, 6.0, 12.0],
            }
        )
        ranked = frame[["feature_x"]].groupby(frame["date"]).rank(pct=True)
        accepted = [
            {
                "candidate_id": "ts_delta_feature_x",
                "op": "expression",
                "features": ["feature_x"],
                "expression_spec": {
                    "op": "ts_delta",
                    "feature": "feature_x",
                    "window": 2,
                },
                "direction": 1,
            }
        ]
        signal = build_accepted_signal(frame, ranked, accepted)
        self.assertEqual(len(signal), len(frame))
        self.assertTrue(signal["prediction"].notna().any())


if __name__ == "__main__":
    unittest.main()
