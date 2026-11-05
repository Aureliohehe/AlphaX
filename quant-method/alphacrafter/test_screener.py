#!/usr/bin/env python3
"""Regression tests for the leakage-safe Screener-lite selection policy."""

from __future__ import annotations

import unittest

import pandas as pd

from miner_lite import Candidate
from screener_lite import select_ensemble


class ScreenerTests(unittest.TestCase):
    def test_trailing_rank_ic_selects_and_weights_factors(self) -> None:
        dates = list(pd.date_range("2020-01-01", periods=11))
        candidates = {
            "factor_a": Candidate("factor_a", "feature", ("feature_a",), ""),
            "factor_b": Candidate("factor_b", "feature", ("feature_b",), ""),
        }
        daily_ics = {
            "factor_a": pd.Series(0.02, index=dates[:10]),
            "factor_b": pd.Series(0.01, index=dates[:10]),
        }
        selected = select_ensemble(
            dates[-1],
            dates,
            candidates,
            daily_ics,
            min_history_days=10,
            min_suitability=0.005,
            similarity_threshold=0.8,
            max_factors=2,
        )
        self.assertEqual([item["candidate_id"] for item in selected], ["factor_a", "factor_b"])
        self.assertAlmostEqual(selected[0]["weight"], 2.0 / 3.0)
        self.assertEqual(selected[0]["direction"], 1)

    def test_current_date_is_not_used_for_suitability(self) -> None:
        dates = list(pd.date_range("2020-01-01", periods=11))
        candidates = {"factor_a": Candidate("factor_a", "feature", ("feature_a",), "")}
        daily_ics = {"factor_a": pd.Series(0.01, index=dates[:10])}
        selected = select_ensemble(
            dates[-1], dates, candidates, daily_ics, 10, 0.02, 0.8, 1
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
