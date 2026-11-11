"""Point-in-time admission, immutable snapshots and replay regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from miner_lite import Candidate
from screener_lite import select_ensemble
from temporal import (
    TEMPORAL_POLICY_VERSION, active_factors_at, label_end_dates, load_timeline,
    purged_mask, session_date, snapshot_date, validate_previous_registry,
    validate_windows, visible_panel, write_snapshot,
)


HERE = Path(__file__).resolve().parent


def fixture_panel(periods=80):
    dates = pd.bdate_range("2021-11-01", periods=periods)
    rng = np.random.default_rng(24)
    rows = []
    for date in dates:
        for asset in range(16):
            value = asset - 7.5
            rows.append({
                "date": date, "ticker": f"S{asset:02d}", "feature_x": value,
                "return_1d": rng.normal(0, .01),
                "target_open_close": .003 * value + rng.normal(0, .01),
                "target_1d": .003 * value + rng.normal(0, .01),
                "target_5d": .005 * value + rng.normal(0, .01),
                "target_open_close_5d": .005 * value + rng.normal(0, .01),
            })
    return pd.DataFrame(rows)


def fixture_snapshot(date, include_factor=True):
    return {
        "temporal_policy_version": TEMPORAL_POLICY_VERSION, "as_of": date,
        "factors": [{
            "candidate_id": "x", "op": "feature", "features": ["feature_x"],
            "hypothesis": "synthetic fixture", "status": "active",
            "created_at": date, "last_validated_at": date,
        }] if include_factor else [],
    }


def run_cli(script, *args, check=True):
    result = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], text=True, capture_output=True)
    if check and result.returncode:
        raise AssertionError(result.stderr + result.stdout)
    return result


class TemporalUnitTests(unittest.TestCase):
    def test_per_asset_label_ends_follow_irregular_sorted_sessions(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-05", "2024-01-02", "2024-01-03", "2024-01-02"]),
            "ticker": ["B", "A", "A", "B"],
        })
        ends = label_end_dates(frame, 1)
        self.assertEqual(ends.iloc[1], pd.Timestamp("2024-01-03"))
        self.assertEqual(ends.iloc[3], pd.Timestamp("2024-01-05"))
        self.assertTrue(pd.isna(ends.iloc[0]))

    def test_purge_is_horizon_specific_at_train_and_asof_boundaries(self):
        frame = fixture_panel(20)
        dates = sorted(frame["date"].unique())
        cutoff = dates[9]
        for horizon in (1, 5):
            ends = label_end_dates(frame, horizon)
            mask = purged_mask(frame["date"], ends, dates[0], dates[-1], cutoff)
            self.assertEqual(int(mask.sum()), (10 - horizon) * 16)
            self.assertTrue((ends.loc[mask] <= cutoff).all())
            at_split = purged_mask(frame["date"], ends, dates[0], cutoff, dates[-1])
            pd.testing.assert_series_equal(mask, at_split)

    def test_truncating_before_shift_censors_precomputed_future_targets(self):
        panel = fixture_panel(20)
        cutoff = sorted(panel["date"].unique())[9]
        visible = visible_panel(panel, cutoff)
        for horizon in (1, 5):
            ends = label_end_dates(visible, horizon)
            self.assertEqual(int(ends.isna().sum()), 16 * horizon)
        self.assertEqual(visible["date"].max(), cutoff)

    def test_dates_windows_and_panel_keys_fail_closed(self):
        for value in (None, "NaT", "2020-01-01T12:00:00", "2020-01-01T00:00:00Z"):
            with self.assertRaises(ValueError):
                session_date(value)
        with self.assertRaises(ValueError):
            validate_windows({"train": ("2020-01-01", "2021-01-01"), "validation": ("2021-01-01", "2022-01-01"), "backtest": ("2023-01-01", "2024-01-01")})
        panel = fixture_panel(2)
        with self.assertRaises(ValueError):
            visible_panel(pd.concat([panel, panel.iloc[:1]]), "2022-01-01")

    def test_snapshot_admission_retirement_and_no_backfill(self):
        timeline = [(pd.Timestamp(date), fixture_snapshot(date, include)) for date, include in (
            ("2024-01-03", True), ("2024-01-05", False), ("2024-01-08", True),
        )]
        self.assertEqual(active_factors_at(timeline, pd.Timestamp("2024-01-02")), [])
        self.assertEqual(len(active_factors_at(timeline, pd.Timestamp("2024-01-03"))), 1)
        self.assertEqual(active_factors_at(timeline, pd.Timestamp("2024-01-06")), [])
        self.assertEqual(len(active_factors_at(timeline, pd.Timestamp("2024-01-08"))), 1)

    def test_snapshot_is_write_once_and_replay_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = fixture_snapshot("2024-01-03")
            path = write_snapshot(directory, payload)
            self.assertEqual(write_snapshot(directory, payload), path)
            original = path.read_text()
            with self.assertRaisesRegex(ValueError, "Immutable"):
                write_snapshot(directory, fixture_snapshot("2024-01-03", False))
            self.assertEqual(path.read_text(), original)
            write_snapshot(directory, fixture_snapshot("2024-01-02", False))
            self.assertEqual(load_timeline(directory)[0][0], pd.Timestamp("2024-01-02"))

    def test_duplicate_snapshot_dates_are_not_resolved_by_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = fixture_snapshot("2024-01-03")
            write_snapshot(directory, payload)
            (directory / "copy.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "same as_of"):
                load_timeline(directory)

    def test_future_registry_legacy_and_future_validation_rejected(self):
        payload = fixture_snapshot("2024-01-03")
        with self.assertRaisesRegex(ValueError, "future registry"):
            validate_previous_registry(payload, "2024-01-02")
        validate_previous_registry(payload, "2024-01-03")
        legacy = {"as_of": "2024-01-01", "factors": []}
        with self.assertRaisesRegex(ValueError, "legacy"):
            snapshot_date(legacy)
        payload["factors"][0]["last_validated_at"] = "2024-01-04"
        with self.assertRaisesRegex(ValueError, "timestamps"):
            snapshot_date(payload)

    def test_delayed_label_must_mature_before_screener_uses_ic(self):
        dates = list(pd.bdate_range("2024-01-01", periods=11))
        candidate = Candidate("x", "feature", ("feature_x",), "")
        ics = {"x": pd.Series(.1, index=dates)}
        maturity = pd.Series(dates[1:], index=dates[:-1])
        selected = select_ensemble(dates[-1], dates, {"x": candidate}, ics, 10, .02, .8, 1, label_available_dates=maturity)
        self.assertEqual(len(selected), 1)
        maturity.iloc[-1] = dates[-1] + pd.Timedelta(days=1)
        selected = select_ensemble(dates[-1], dates, {"x": candidate}, ics, 10, .02, .8, 1, label_available_dates=maturity)
        self.assertEqual(selected, [])


class TemporalIntegrationTests(unittest.TestCase):
    def run_miner(self, root, input_name, output_name, *extra):
        candidates = root / "candidates.json"
        candidates.write_text(json.dumps([{
            "candidate_id": "x", "op": "feature", "features": ["feature_x"], "hypothesis": "synthetic fixture",
        }]))
        return run_cli("miner_lite.py",
            "--input", root / input_name, "--output-dir", root / output_name,
            "--candidate-file", candidates, "--as-of", "2022-01-14",
            "--train-start", "2021-11-01", "--train-end", "2021-12-03",
            "--validation-start", "2021-12-06", "--validation-end", "2022-01-14",
            "--backtest-start", "2022-01-17", "--backtest-end", "2022-02-28", *extra)

    def test_miner_future_mutation_does_not_change_admission_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = fixture_panel()
            panel.to_csv(root / "original.csv", index=False)
            changed = panel.copy()
            # Deliberately overwrite future rows and all labels straddling either
            # segment end, including targets physically stored on past rows.
            changed.loc[changed["date"] > "2022-01-14", "feature_x"] *= -1000
            for column, horizon in (("target_open_close", 1), ("target_1d", 1), ("target_5d", 5), ("target_open_close_5d", 5)):
                ends = label_end_dates(changed, horizon)
                bad = ((changed["date"] <= "2021-12-03") & (ends > "2021-12-03")) | (ends > "2022-01-14")
                changed.loc[bad, column] = 1000.
            changed.to_csv(root / "changed.csv", index=False)
            self.run_miner(root, "original.csv", "a")
            self.run_miner(root, "changed.csv", "b")
            for name in ("candidate_metrics.json", "factor_library.json", "predictions.csv"):
                self.assertEqual((root / "a" / name).read_text(), (root / "b" / name).read_text(), name)
            library = json.loads((root / "a/factor_library.json").read_text())
            self.assertEqual(len(library["factors"]), 1)  # unchanged acceptance, not an empty-library tautology
            signals = pd.read_csv(root / "a/predictions.csv", parse_dates=["datetime"])
            self.assertEqual(set(signals["datetime"]), {pd.Timestamp("2022-01-14")})
            audit = json.loads((root / "a/temporal_audit.json").read_text())
            self.assertEqual(audit["segments"]["5"]["train"]["purged_rows"], 80)
            self.assertEqual(audit["segments"]["1"]["validation"]["purged_rows"], 16)
            self.assertEqual(audit["segments"]["5"]["backtest"]["eligible_rows"], 0)
            self.assertTrue((root / "a/snapshots/2022-01-14.json").exists())

    def test_miner_does_not_reuse_a_future_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_panel().to_csv(root / "features.csv", index=False)
            (root / "registry.json").write_text(json.dumps(fixture_snapshot("2025-01-01")))
            result = run_cli("miner_lite.py", "--input", root / "features.csv", "--registry", root / "registry.json",
                             "--output-dir", root / "out", "--as-of", "2022-01-14", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("future registry", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_screener_replays_activation_empty_snapshot_and_readmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = fixture_panel(22)
            panel.to_csv(root / "features.csv", index=False)
            dates = sorted(panel["date"].unique())
            for index, include in ((12, True), (15, False), (18, True)):
                date = str(pd.Timestamp(dates[index]).date())
                write_snapshot(root / "snapshots", fixture_snapshot(date, include))
            run_cli("screener_lite.py", "--input", root / "features.csv", "--library-history", root / "snapshots", "--output-dir", root / "out")
            regime = pd.read_csv(root / "out/regime.csv")
            self.assertEqual(regime["active_factor_count"].tolist(), [0]*12 + [1]*3 + [0]*3 + [1]*4)
            signals = pd.read_csv(root / "out/predictions.csv", parse_dates=["datetime"])
            self.assertTrue((signals["datetime"] >= dates[12]).all())
            self.assertFalse(signals["datetime"].between(dates[15], dates[17]).any())
            self.assertTrue((signals["datetime"] >= dates[18]).any())
            # Looking only up to the retirement cutoff must produce exactly the
            # same earlier predictions even though later readmission exists.
            run_cli("screener_lite.py", "--input", root / "features.csv", "--library-history", root / "snapshots", "--output-dir", root / "early", "--as-of", str(pd.Timestamp(dates[15]).date()))
            early = pd.read_csv(root / "early/predictions.csv", parse_dates=["datetime"])
            pd.testing.assert_frame_equal(early, signals[signals["datetime"] <= dates[15]].reset_index(drop=True))
            # Restrict output only, not the ten-day history needed for suitability.
            run_cli("screener_lite.py", "--input", root / "features.csv", "--library-history", root / "snapshots", "--output-dir", root / "late", "--signal-start", str(pd.Timestamp(dates[18]).date()))
            late = pd.read_csv(root / "late/predictions.csv", parse_dates=["datetime"])
            pd.testing.assert_frame_equal(late, signals[signals["datetime"] >= dates[18]].reset_index(drop=True))

    def test_irregular_label_maturity_is_prefix_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = fixture_panel(25)
            dates = sorted(panel["date"].unique())
            panel = panel[~((panel["ticker"] == "S00") & panel["date"].between(dates[14], dates[17]))]
            panel.to_csv(root / "features.csv", index=False)
            snapshot = fixture_snapshot(str(pd.Timestamp(dates[10]).date()))
            (root / "library.json").write_text(json.dumps(snapshot))
            args = ("--input", root / "features.csv", "--factor-library", root / "library.json")
            run_cli("screener_lite.py", *args, "--output-dir", root / "full")
            run_cli("screener_lite.py", *args, "--output-dir", root / "early", "--as-of", str(pd.Timestamp(dates[16]).date()))
            full = pd.read_csv(root / "full/predictions.csv", parse_dates=["datetime"])
            early = pd.read_csv(root / "early/predictions.csv", parse_dates=["datetime"])
            self.assertFalse(early.empty)
            pd.testing.assert_frame_equal(early, full[full["datetime"] <= dates[16]].reset_index(drop=True))
            self.assertFalse(early["datetime"].between(dates[14], dates[16]).any())

    def test_screener_future_static_snapshot_emits_no_historical_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_panel(22).to_csv(root / "features.csv", index=False)
            (root / "library.json").write_text(json.dumps(fixture_snapshot("2025-01-01")))
            run_cli("screener_lite.py", "--input", root / "features.csv", "--factor-library", root / "library.json", "--output-dir", root / "out")
            self.assertTrue(pd.read_csv(root / "out/predictions.csv").empty)
            self.assertTrue(pd.read_csv(root / "out/selection.csv").empty)

    def test_legacy_requires_explicit_diagnostic_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_panel(22).to_csv(root / "features.csv", index=False)
            payload = fixture_snapshot("2025-01-01")
            del payload["temporal_policy_version"]
            (root / "library.json").write_text(json.dumps(payload))
            args = ("--input", root / "features.csv", "--factor-library", root / "library.json", "--output-dir", root / "out")
            result = run_cli("screener_lite.py", *args, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("legacy", result.stderr)
            run_cli("screener_lite.py", *args, "--allow-retrospective-library")
            summary = json.loads((root / "out/registry.json").read_text())
            self.assertEqual(summary["library_scope"], "retrospective_diagnostic")
            self.assertFalse(pd.read_csv(root / "out/predictions.csv").empty)


if __name__ == "__main__":
    unittest.main()
