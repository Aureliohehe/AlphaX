#!/usr/bin/env python3
"""Regression tests for the leakage-safe Screener-lite selection policy."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from miner_lite import Candidate
from screener_lite import assess_regime_with_llm, ensemble_signal, select_ensemble
from temporal import TEMPORAL_POLICY_VERSION


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

    def test_missing_component_is_not_filled_and_order_is_irrelevant(self) -> None:
        scores = {"a": pd.Series([np.nan, .4]), "b": pd.Series([.2, .6])}
        factors = [{"candidate_id": name, "direction": 1, "weight": .5} for name in scores]
        signal = ensemble_signal(scores, factors, pd.RangeIndex(2))
        self.assertTrue(pd.isna(signal.iloc[0]))
        self.assertAlmostEqual(signal.iloc[1], .5)
        pd.testing.assert_series_equal(signal, ensemble_signal(scores, factors[::-1], pd.RangeIndex(2)))

    def test_regime_uses_sessions_not_last_sixty_stocks(self) -> None:
        frame = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")] * 100,
            "return_1d": [-.1] * 50 + [.1] * 50,
        })
        reply = json.dumps({"regime": "mixed_or_sideways", "confidence": .5})
        with patch("screener_lite._deepseek_completion", return_value=reply) as call:
            assess_regime_with_llm(frame, pd.Timestamp("2024-01-02"), "deepseek-chat")
        prompt = call.call_args[0][0][1]["content"]
        self.assertIn('"return_1d": 0.0', prompt)
        self.assertIn("breadth: 0.5", prompt)

    def test_feature_builder_preserves_unlabelled_latest_session(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=180)
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
        raw = pd.DataFrame({
            "date": dates, "ticker": "A", "open": close * .999,
            "high": close * 1.01, "low": close * .99, "close": close,
            "adj_close": close, "volume": rng.integers(1000, 10000, len(dates)),
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            raw.to_csv(root / "raw/A.csv", index=False)
            builder = Path(__file__).resolve().parents[1] / "machine-learn/build_features.py"
            subprocess.run([
                sys.executable, str(builder), "--input", str(root / "raw"),
                "--output", str(root / "features.csv.gz"),
            ], check=True, capture_output=True, text=True)
            frame = pd.read_csv(root / "features.csv.gz", parse_dates=["date"])
            self.assertEqual(frame["date"].iloc[-1], dates[-1])
            self.assertTrue(pd.isna(frame["target_open_close"].iloc[-1]))
            self.assertTrue(pd.isna(frame["target_1d"].iloc[-1]))

    def test_cli_keeps_latest_unlabelled_cross_section(self) -> None:
        dates = pd.date_range("2020-01-01", periods=12)
        frame = pd.DataFrame({
            "date": np.repeat(dates, 3), "ticker": ["A", "B", "C"] * 12,
            "feature_x": [1., 2., 3.] * 12, "return_1d": [-.01, 0., .01] * 12,
            "target_open_close": [-.01, 0., .01] * 11 + [np.nan] * 3,
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame.to_csv(root / "features.csv", index=False)
            (root / "library.json").write_text(json.dumps({
                "temporal_policy_version": TEMPORAL_POLICY_VERSION, "as_of": "2020-01-01",
                "factors": [{
                    "candidate_id": "x", "op": "feature", "features": ["feature_x"], "hypothesis": "test",
                    "created_at": "2020-01-01", "last_validated_at": "2020-01-01",
                }],
            }))
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("screener_lite.py")),
                "--input", str(root / "features.csv"), "--factor-library", str(root / "library.json"),
                "--output-dir", str(root / "out"),
            ], check=True, capture_output=True, text=True)
            signals = pd.read_csv(root / "out/predictions.csv", parse_dates=["datetime"])
            latest = signals[signals["datetime"] == dates[-1]]
            self.assertEqual(len(latest), 3)
            self.assertEqual(latest["instrument"].tolist(), ["A", "B", "C"])

    def test_current_date_is_not_used_for_suitability(self) -> None:
        dates = list(pd.date_range("2020-01-01", periods=11))
        candidates = {"factor_a": Candidate("factor_a", "feature", ("feature_a",), "")}
        daily_ics = {"factor_a": pd.Series([.01] * 10 + [1.], index=dates)}
        selected = select_ensemble(
            dates[-1], dates, candidates, daily_ics, 10, 0.02, 0.8, 1
        )
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
