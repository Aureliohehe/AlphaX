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
from temporal import TEMPORAL_POLICY_VERSION, META_COLUMNS, label_end_dates, purged_mask, visible_panel
from data_contract import (
    calendar_audit, forward_labels, is_feature_column, read_calendar,
    resolve_label_ends,
)
from build_features import add_features
from train_xgboost_qlib import load_frame
from run_alpha158_model_diagnostic import candidate_features, fold_masks, daily_metrics
from audit_sp500_universe import audit as audit_sp500_universe, snapshot_entries
from compare_alpha158_qlib import compare as compare_qlib_fields
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

    def test_snapshot_membership_audit_detects_pre_entry_and_never_certifies_pit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.bdate_range("2022-12-26", periods=4)
            pd.DataFrame({"Symbol": ["A", "B"], "Date added": [dates[2], dates[0]]}).to_csv(root / "members.csv", index=False)
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            frame = pd.DataFrame({"date": list(dates) * 2, "ticker": ["A"] * 4 + ["B"] * 4})
            frame.to_csv(root / "features.csv", index=False)
            result = audit_sp500_universe(root / "features.csv", root / "members.csv", root / "calendar.csv")
            self.assertEqual(result["feature_rows_before_current_snapshot_date_added"], 2)
            self.assertFalse(result["historical_membership_complete"])
            self.assertEqual(result["status"], "not_point_in_time_certified")
            self.assertTrue(result["calendar_audit_after_feature_warmup"]["matches_reference"])
            pd.DataFrame({"Symbol": ["A", "A"], "Date added": [dates[0], dates[1]]}).to_csv(root / "members.csv", index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                snapshot_entries(root / "members.csv")

    def test_reference_requires_unique_sorted_dates_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.csv"
            for dates in (["2024-01-02", "2024-01-02"], ["2024-01-03", "2024-01-02"], ["2024-01-02T12:00:00"]):
                pd.DataFrame({"date": dates}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    read_calendar(path)
        with self.assertRaisesRegex(ValueError, "cover"):
            calendar_audit(pd.Series(pd.bdate_range("2024-01-01", periods=10)), pd.bdate_range("2024-01-01", periods=5), "2024-01-01", "2024-01-20")

    def test_pipeline_warmup_allowance_only_exempts_contiguous_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = explicit_panel(70)
            dates = sorted(panel["date"].unique())
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            date = lambda index: str(pd.Timestamp(dates[index]).date())
            args = [
                "--input", root / "features.csv", "--calendar", root / "calendar.csv",
                "--require-explicit-label-dates", "--train-start", date(0),
                "--initial-as-of", date(44), "--admission-sessions", "20",
                "--min-train-sessions", "20", "--selection-start", date(45),
                "--selection-end", date(54), "--holdout-start", date(55),
                "--holdout-end", date(64), "--output-dir", root / "dry", "--dry-run",
            ]
            panel.loc[panel["date"] >= dates[2]].to_csv(root / "features.csv", index=False)
            failed = run_cli("run_pipeline.py", *args, check=False)
            self.assertIn("Feature calendar mismatch", failed.stderr)
            failed = run_cli("run_pipeline.py", *args, "--allow-feature-warmup-sessions", "1", check=False)
            self.assertIn("Feature calendar mismatch", failed.stderr)
            success = run_cli("run_pipeline.py", *args, "--allow-feature-warmup-sessions", "2")
            audit = json.loads(success.stdout)["calendar_audit"]
            self.assertEqual(audit["feature_warmup_sessions"], 2)
            self.assertEqual(audit["first_feature_session"], date(2))
            self.assertFalse(audit["matches_reference"])
            self.assertTrue(audit["matches_reference_after_warmup"])
            panel.loc[(panel["date"] >= dates[2]) & (panel["date"] != dates[50])].to_csv(root / "features.csv", index=False)
            failed = run_cli("run_pipeline.py", *args, "--allow-feature-warmup-sessions", "2", check=False)
            self.assertIn("Feature calendar mismatch", failed.stderr)
            self.assertFalse((root / "dry").exists())

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


class ModelDiagnosticTests(unittest.TestCase):
    def test_curated_features_are_frozen_and_raw(self):
        source = HERE / "candidates_alpha158_curated_v1.json"
        features = candidate_features(source)
        self.assertEqual(len(features), 18)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            candidates = json.loads(source.read_text())
            candidates[0]["expression"] = {"op": "rank", "input": {"op": "feature", "feature": features[0]}}
            path.write_text(json.dumps(candidates))
            with self.assertRaisesRegex(ValueError, "raw, single-column"):
                candidate_features(path)

    def test_model_fold_purges_unsettled_labels_and_does_not_fit_validation(self):
        frame = explicit_panel(45).copy()
        dates = sorted(frame["date"].unique())
        # Fit must not use the last training signal whose next-day target
        # settles after training ends; validation's final signal also cannot
        # enter reported IC, though it is still predicted.
        iso = lambda i: str(pd.Timestamp(dates[i]).date())
        train, valid, scored = fold_masks(frame, iso(19), iso(20), iso(29))
        self.assertFalse(train[frame["date"] == dates[19]].any())
        self.assertTrue(valid[frame["date"] == dates[29]].all())
        self.assertFalse(scored[frame["date"] == dates[29]].any())
        changed = frame.copy()
        changed.loc[valid, "target_open_close"] = -999.
        again, _, _ = fold_masks(changed, iso(19), iso(20), iso(29))
        pd.testing.assert_series_equal(train, again)
        # No returned IC may consume unlabelled/unsettled rows.
        summary, daily = daily_metrics(frame, frame["feature_x"], scored)
        self.assertEqual(summary["rows"], int(scored.sum()))
        self.assertEqual(daily["date"].max(), iso(28))


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

    def test_qlib_comparator_flags_value_and_missingness_mismatch(self):
        index = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2022-01-03")), ("A", pd.Timestamp("2022-01-04"))], names=["ticker", "date"])
        local = pd.DataFrame({"alpha158_kmid2": [1.0, np.nan]}, index=index)
        official = pd.DataFrame({"alpha158_kmid2": [1.1, 2.]}, index=index)
        metric = compare_qlib_fields(local, official, ["alpha158_kmid2"], atol=1e-4)["alpha158_kmid2"]
        self.assertEqual(metric["samples"], 1)
        self.assertEqual(metric["missing_mismatch"], 1)
        self.assertEqual(metric["rows_exceeding_tolerance"], 1)

    def test_miner_paper_label_mode_is_explicit_and_cannot_inherit_execution_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = explicit_panel(70)
            dates = sorted(panel["date"].unique())
            panel.to_csv(root / "features.csv", index=False)
            (root / "candidates.json").write_text(json.dumps([{
                "candidate_id": "x", "op": "feature", "features": ["feature_x"], "hypothesis": "fixture",
            }]))
            date = lambda index: str(pd.Timestamp(dates[index]).date())
            args = ["--input", root / "features.csv", "--candidate-file", root / "candidates.json",
                    "--require-explicit-label-dates", "--as-of", date(59),
                    "--train-start", date(0), "--train-end", date(29),
                    "--validation-start", date(30), "--validation-end", date(59),
                    "--backtest-start", date(60), "--backtest-end", date(60)]
            run_cli("miner_lite.py", *args, "--output-dir", root / "execution")
            self.assertEqual(json.loads((root / "execution/registry.json").read_text())["selection"]["target_sources"]["1d"], "target_open_close")
            run_cli("miner_lite.py", *args, "--label-mode", "paper_close", "--output-dir", root / "paper")
            sources = json.loads((root / "paper/registry.json").read_text())["selection"]["target_sources"]
            self.assertEqual(sources, {"1d": "target_1d", "5d": "target_5d"})
            snapshot = json.loads((root / "paper/snapshots" / (date(59) + ".json")).read_text())
            self.assertEqual(snapshot["temporal_audit"]["label_mode"], "paper_close")
            failed_screen = run_cli("screener_lite.py", "--input", root / "features.csv",
                                    "--library-history", root / "paper/snapshots", "--as-of", date(59),
                                    "--output-dir", root / "screen", check=False)
            self.assertIn("diagnostic only", failed_screen.stderr)
            failed = run_cli("miner_lite.py", *args, "--label-mode", "paper_close",
                             "--registry", root / "execution/registry.json", "--output-dir", root / "mismatch", check=False)
            self.assertIn("different label targets", failed.stderr)

    def test_execution_demo_harness_creates_real_positions_on_synthetic_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = explicit_panel(50)
            dates = sorted(panel["date"].unique())
            date = lambda i: str(pd.Timestamp(dates[i]).date())
            panel.to_csv(root / "features.csv", index=False)
            pd.DataFrame({"date": dates}).to_csv(root / "calendar.csv", index=False)
            miner = root / "miner"
            miner.mkdir()
            (miner / "registry.json").write_text(json.dumps({"as_of": date(20), "generator": "llm_cluster:fixture:1", "accepted_count": 0}))
            (miner / "candidates.json").write_text(json.dumps([{
                "candidate_id": "fixture_factor", "features": ["feature_x"],
                "expression": {"op": "feature", "feature": "feature_x"}, "hypothesis": "synthetic only",
            }]))
            metric = {"mean_daily_ic": .01, "mean_daily_rank_ic": .02, "icir": .1, "rank_icir": .1,
                      "rank_ic_hit_ratio": .55, "coverage": 1., "turnover": .2}
            (miner / "candidate_metrics.json").write_text(json.dumps([{
                "candidate_id": "fixture_factor", "direction_from_train": 1,
                "gate_results": {"train_1d": False, "validation_1d": False, "train_5d": False, "validation_5d": False},
                "horizons": {"1": {"train": metric, "validation": metric}},
            }]))
            (miner / "llm_candidate_audit.json").write_text("[]")
            (miner / "temporal_audit.json").write_text(json.dumps({"label_mode": "execution"}))
            run_cli("run_agent_execution_demo.py", "--miner-dir", miner,
                    "--input", root / "features.csv", "--calendar", root / "calendar.csv",
                    "--admission-as-of", date(20), "--selection-start", date(21), "--selection-end", date(35),
                    "--output-dir", root / "demo")
            summary = json.loads((root / "demo/summary.json").read_text())
            self.assertEqual(summary["status"], "completed_exploratory_demo")
            self.assertEqual(summary["library_scope"], "retrospective_diagnostic")
            library = json.loads((root / "demo/factor_library_demo.json").read_text())
            self.assertEqual(library["library_scope"], "retrospective_diagnostic")
            self.assertNotEqual(library["factors"][0]["created_at_utc"][:10], date(20))
            self.assertFalse((root / "demo/snapshots").exists())
            self.assertGreater(summary["invested_days"], 0)
            self.assertGreater(summary["position_rows"], 0)
            self.assertTrue((pd.read_csv(root / "demo/trades_daily.csv")["transaction_cost"] > 0).any())

    def test_strict_miner_refuses_legacy_registry_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit_panel().to_csv(root / "features.csv", index=False)
            (root / "registry.json").write_text(json.dumps({
                "temporal_policy_version": TEMPORAL_POLICY_VERSION, "as_of": "2021-11-01", "accepted_factors": [],
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
