"""Explicit raw-label dates, reference calendars and metadata leakage tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# temporal installs the sibling standalone-script directory on sys.path.
from temporal import META_COLUMNS, label_end_dates, purged_mask, visible_panel
from data_contract import (
    calendar_audit, forward_labels, is_feature_column, read_calendar,
    resolve_label_ends,
)
from build_features import add_features
from train_xgboost_qlib import load_frame
from trader_lite import TraderConfig, load_inputs, simulate
from test_temporal import fixture_panel, run_cli


HERE = Path(__file__).resolve().parent


def raw_panel(periods=180):
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
    return pd.DataFrame({
        "date": dates, "ticker": "A", "open": close * .999,
        "high": close * 1.01, "low": close * .99, "close": close,
        "adj_close": close, "volume": rng.integers(1000, 10000, len(dates)),
    })


def explicit_panel(periods=100):
    panel = fixture_panel(periods)
    for horizon, targets in ((1, ("target_1d", "target_open_close")), (5, ("target_5d", "target_open_close_5d"))):
        ends = label_end_dates(panel, horizon)
        panel[f"label_end_{horizon}d"] = ends
        panel.loc[ends.isna(), list(targets)] = np.nan
    return panel


class LabelDateTests(unittest.TestCase):
    def test_builder_dates_and_prices_align_before_feature_filtering(self):
        raw = raw_panel()
        result = add_features(raw)
        self.assertEqual(result["label_end_1d"].iloc[0], raw["date"].iloc[1])
        self.assertEqual(result["label_end_5d"].iloc[0], raw["date"].iloc[5])
        self.assertAlmostEqual(result["target_open_close_5d"].iloc[0], raw["close"].iloc[5] / raw["open"].iloc[1] - 1)
        self.assertTrue(result["label_end_5d"].tail(5).isna().all())
        # Delete an intermediate feature row; the raw target date must not move.
        filtered = result.drop(index=[1])
        ends = resolve_label_ends(filtered, 1, require_explicit=True)
        self.assertEqual(ends.iloc[0], raw["date"].iloc[1])
        legacy = resolve_label_ends(filtered.drop(columns=["label_end_1d"]), 1)
        self.assertEqual(legacy.iloc[0], raw["date"].iloc[2])

    def test_reference_calendar_does_not_shift_to_next_available_stock_quote(self):
        calendar = pd.bdate_range("2024-01-01", periods=8)
        dates = pd.Series(calendar.delete(1))
        close = pd.Series(np.arange(len(dates), dtype=float) + 100)
        labels = forward_labels(dates, close, close, calendar)
        self.assertEqual(labels["label_end_1d"].iloc[0], calendar[1])
        self.assertTrue(pd.isna(labels["target_1d"].iloc[0]))
        self.assertTrue(pd.isna(labels["target_open_close_5d"].iloc[0]))
        self.assertEqual(labels["label_end_5d"].iloc[0], calendar[5])
        self.assertAlmostEqual(labels["target_5d"].iloc[0], 104 / 100 - 1)
        # The no-calendar path preserves the original next-observation horizon.
        legacy = forward_labels(dates, close, close)
        self.assertEqual(legacy["label_end_1d"].iloc[0], calendar[2])
        self.assertAlmostEqual(legacy["target_1d"].iloc[0], .01)

    def test_explicit_maturity_survives_asof_truncation_and_purge(self):
        panel = explicit_panel(20)
        cutoff = sorted(panel["date"].unique())[9]
        visible = visible_panel(panel, cutoff)
        ends = resolve_label_ends(visible, 5, True)
        self.assertTrue((ends[visible["date"] == cutoff] > cutoff).all())
        mask = purged_mask(visible["date"], ends, visible["date"].min(), cutoff, cutoff)
        self.assertEqual(int(mask.sum()), 5 * 16)

    def test_bad_metadata_fails_closed_and_never_falls_back_per_row(self):
        panel = explicit_panel(10).iloc[:16].copy()
        for bad in (None, "not a date", "2021-11-01", "2021-11-02T01:00:00", "2021-11-02T00:00:00Z"):
            changed = panel.copy()
            changed["label_end_1d"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                resolve_label_ends(changed, 1)
        changed = panel.copy()
        changed["label_end_5d"] = changed["label_end_1d"]
        with self.assertRaisesRegex(ValueError, "Five-session"):
            resolve_label_ends(changed, 5)
        with self.assertRaisesRegex(ValueError, "strict label-date"):
            resolve_label_ends(panel.drop(columns=["label_end_1d"]), 1, True)

    def test_reference_rejects_observation_horizons_that_skip_market_sessions(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
        frame = pd.DataFrame({"date": dates, "ticker": "A", "label_end_1d": [dates[1], pd.NaT]})
        with self.assertRaisesRegex(ValueError, "reference market sessions"):
            resolve_label_ends(frame, 1, True, pd.bdate_range("2024-01-01", periods=6))

    def test_label_metadata_is_excluded_from_all_model_features(self):
        self.assertIn("label_end_1d", META_COLUMNS)
        self.assertIn("label_end_5d", META_COLUMNS)
        for name in ("label_end_1d", "label_end_5d", "label_end_extra", "target_extra"):
            self.assertFalse(is_feature_column(name))
        with tempfile.TemporaryDirectory() as tmp:
            panel = explicit_panel(10)
            panel["open_close_return"] = .01
            panel["split"] = "train"
            panel["label_end_extra"] = "2099-01-01"
            path = Path(tmp) / "features.csv"
            panel.to_csv(path, index=False)
            _, columns = load_frame(path)
            self.assertFalse(any(name.startswith(("label_end_", "target_")) for name in columns))
            self.assertIn("feature_x", columns)

    def test_raw_duplicates_are_rejected_not_silently_deduplicated(self):
        raw = raw_panel(10)
        with self.assertRaisesRegex(ValueError, "unique"):
            add_features(pd.concat([raw, raw.iloc[:1]]))


class CalendarTests(unittest.TestCase):
    def test_audit_detects_missing_entire_sessions_and_non_sessions(self):
        reference = pd.bdate_range("2024-01-01", periods=10)
        observed = pd.Series(reference.delete(3).tolist() + [pd.Timestamp("2024-01-06")])
        audit = calendar_audit(observed, reference, reference[0], reference[-1])
        self.assertEqual(audit["missing_sessions"], ["2024-01-04"])
        self.assertEqual(audit["unexpected_sessions"], ["2024-01-06"])
        self.assertFalse(audit["matches_reference"])

    def test_reference_requires_unique_sorted_dates_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.csv"
            for dates in (["2024-01-02", "2024-01-02"], ["2024-01-03", "2024-01-02"], ["2024-01-02T12:00:00"]):
                pd.DataFrame({"date": dates}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    read_calendar(path)
        with self.assertRaisesRegex(ValueError, "cover"):
            calendar_audit(pd.Series(pd.bdate_range("2024-01-01", periods=10)), pd.bdate_range("2024-01-01", periods=5), "2024-01-01", "2024-01-20")

    def test_exact_end_and_calendar_allow_return_despite_filtered_execution_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2024-01-01", periods=4)
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            # Features for execution day Jan 2 were filtered out, but its price
            # and the matching raw label were recorded when building the panel.
            frame = pd.DataFrame({
                "date": [dates[0], dates[2]], "ticker": ["A", "A"],
                "label_end_1d": [dates[1], dates[3]], "target_open_close": [.02, .01],
            })
            frame.to_csv(root / "features.csv", index=False)
            pd.DataFrame({"datetime": [dates[0]], "instrument": ["A"], "prediction": [1.]}).to_csv(root / "predictions.csv", index=False)
            loaded = load_inputs(root / "predictions.csv", root / "features.csv", as_of="2024-01-02", calendar_path=root / "calendar.csv", require_explicit_label_dates=True)
            self.assertEqual(loaded["execution_date"].iloc[0], dates[1])
            report, _ = simulate(loaded, TraderConfig(.8, 1., 1, 0), "2024-01-01", "2024-01-02", 0.)
            self.assertAlmostEqual(report["net_return"].iloc[0], .016)
            # The pipeline will still reject the missing full feature session:
            # a valid stored P&L is not evidence of complete signal coverage.
            audit = calendar_audit(frame["date"], pd.DatetimeIndex(dates), dates[0], dates[-1])
            self.assertFalse(audit["matches_reference"])


class DataContractIntegrationTests(unittest.TestCase):
    def test_builder_exports_dates_calendar_manifest_and_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            raw = raw_panel()
            raw.to_csv(root / "raw/A.csv", index=False)
            output = root / "features.csv.gz"
            command = [sys.executable, str(HERE.parent / "machine-learn/build_features.py"), "--input", str(root / "raw"), "--output", str(output)]
            subprocess.run(command, capture_output=True, text=True, check=True)
            panel = pd.read_csv(output, parse_dates=["date", "label_end_1d", "label_end_5d"])
            self.assertEqual(panel["date"].iloc[-1], raw["date"].iloc[-1])
            first_raw_position = raw.index[raw["date"] == panel["date"].iloc[0]][0]
            self.assertEqual(panel["label_end_5d"].iloc[0], raw["date"].iloc[first_raw_position + 5])
            manifest = json.loads((root / "features.csv.gz.manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "features-with-label-dates-v1")
            self.assertEqual(manifest["calendar_kind"], "observed_raw_union_not_exchange_verified")
            self.assertEqual(len(read_calendar(root / "features.csv.gz.calendar.csv")), len(raw))
            self.assertNotIn("label_end_1d", manifest["feature_columns"])
            prior = output.read_bytes()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(prior, output.read_bytes())

    def test_strict_miner_refuses_legacy_registry_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_panel().to_csv(root / "features.csv", index=False)
            (root / "registry.json").write_text(json.dumps({
                "temporal_policy_version": "label-end-v1", "as_of": "2021-11-01", "accepted_factors": [],
            }))
            result = run_cli("miner_lite.py", "--input", root / "features.csv", "--registry", root / "registry.json",
                             "--output-dir", root / "out", "--as-of", "2022-01-14", "--require-explicit-label-dates", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("legacy/unknown label-date provenance", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_strict_pipeline_and_calendar_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = explicit_panel(70)
            dates = sorted(panel["date"].unique())
            panel.to_csv(root / "features.csv", index=False)
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            (root / "candidates.json").write_text(json.dumps([{
                "candidate_id": "x", "op": "feature", "features": ["feature_x"], "hypothesis": "fixed fixture",
            }]))
            date = lambda index: str(pd.Timestamp(dates[index]).date())
            args = [
                "--input", root / "features.csv", "--candidate-file", root / "candidates.json",
                "--calendar", root / "calendar.csv", "--require-explicit-label-dates",
                "--train-start", date(0), "--initial-as-of", date(39),
                "--admission-sessions", "20", "--min-train-sessions", "20",
                "--selection-start", date(40), "--selection-end", date(49),
                "--holdout-start", date(50), "--holdout-end", date(59), "--configs", ".8,0,2,2",
            ]
            run_cli("run_pipeline.py", *args, "--output-dir", root / "ok")
            summary = json.loads((root / "ok/summary.json").read_text())
            self.assertTrue(summary["calendar_audit"]["matches_reference"])
            self.assertEqual(summary["label_end_source"], "explicit_raw_label_dates")
            self.assertGreater(summary["holdout_invested_days"], 0)
            audit = json.loads((root / "ok/miner" / date(39) / "temporal_audit.json").read_text())
            self.assertEqual(audit["label_end_source"]["5"], "explicit_raw_label_dates")
            frozen = (root / "ok/trader/selection.json").read_text()
            changed = panel.copy()
            for horizon, targets in ((1, ("target_1d", "target_open_close")), (5, ("target_5d", "target_open_close_5d"))):
                changed.loc[changed[f"label_end_{horizon}d"] > dates[49], list(targets)] *= -10
            changed.loc[changed["date"] > dates[49], "feature_x"] *= -3
            changed.to_csv(root / "features.csv", index=False)
            run_cli("run_pipeline.py", *args, "--output-dir", root / "changed")
            self.assertEqual((root / "changed/trader/selection.json").read_text(), frozen)
            # Complete-market date missingness must fail even on dry run.
            panel.loc[panel["date"] != dates[45]].to_csv(root / "features.csv", index=False)
            failed = run_cli("run_pipeline.py", *args, "--output-dir", root / "bad", "--dry-run", check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("calendar mismatch", failed.stderr)
            self.assertFalse((root / "bad").exists())
            panel.drop(columns=["label_end_1d", "label_end_5d"]).to_csv(root / "features.csv", index=False)
            failed = run_cli("run_pipeline.py", *args, "--output-dir", root / "legacy", "--dry-run", check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Strict label-date", failed.stderr)
            self.assertFalse((root / "legacy").exists())


if __name__ == "__main__":
    unittest.main()
