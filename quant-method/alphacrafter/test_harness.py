#!/usr/bin/env python3
"""Small regression tests for Miner-lite temporal and cost boundaries."""

from __future__ import annotations

import unittest

import pandas as pd

from miner_lite import Candidate, META_COLUMNS
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


if __name__ == "__main__":
    unittest.main()
