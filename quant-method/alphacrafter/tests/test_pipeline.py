"""End-to-end schedule, immutable selection, failure and temporal tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from run_pipeline import Pipeline, build_plan, fingerprint_set, verify_fingerprints
from .test_temporal import fixture_panel, run_cli
from trader_lite import load_inputs, simulate, TraderConfig


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.calendar = list(pd.bdate_range("2021-01-01", periods=200))
        self.kwargs = dict(
            initial_as_of=str(self.calendar[49].date()), train_start="2021-01-01",
            selection_start=str(self.calendar[50].date()), selection_end=str(self.calendar[89].date()),
            holdout_start=str(self.calendar[90].date()), holdout_end=str(self.calendar[139].date()),
            maintenance_days=14, admission_sessions=20, min_train_sessions=20,
        )

    def test_windows_move_and_maintenance_waits_for_first_session_due(self):
        plan = build_plan(self.calendar, **self.kwargs)
        rounds = plan["rounds"]
        self.assertGreater(len(rounds), 3)
        for index, row in enumerate(rounds):
            self.assertEqual(row["observed_admission_sessions"], 20)
            self.assertLess(row["train_end"], row["validation_start"])
            self.assertEqual(row["validation_end"], row["as_of"])
            self.assertEqual(row["phase"], "selection" if row["as_of"] <= self.kwargs["selection_end"] else "holdout")
            if index:
                due = pd.Timestamp(rounds[index - 1]["as_of"]) + pd.Timedelta(days=14)
                expected = next(date for date in self.calendar if date >= due)
                self.assertEqual(pd.Timestamp(row["as_of"]), expected)
                self.assertGreater(row["train_end"], rounds[index - 1]["train_end"])

    def test_non_session_initial_cutoff_uses_previous_observed_session(self):
        friday = next(day for day in self.calendar[40:49] if day.weekday() == 4)
        plan = build_plan(self.calendar, **{**self.kwargs, "initial_as_of": str((friday + pd.Timedelta(days=2)).date())})
        self.assertEqual(plan["initial_as_of_effective"], str(friday.date()))

    def test_initial_maintenance_prefix_does_not_depend_on_future_sessions(self):
        full = build_plan(self.calendar, **self.kwargs)
        changed_future = self.calendar[:120] + [date + pd.Timedelta(days=2) for date in self.calendar[120:]]
        changed = build_plan(changed_future, **self.kwargs)
        self.assertEqual([row for row in full["rounds"] if row["phase"] == "selection"],
                         [row for row in changed["rounds"] if row["phase"] == "selection"])

    def test_rejects_overlaps_insufficient_history_and_truncated_data(self):
        invalid = [
            {"initial_as_of": self.kwargs["selection_start"]},
            {"holdout_start": self.kwargs["selection_end"]},
            {"admission_sessions": 100}, {"admission_sessions": 5},
            {"maintenance_days": 0}, {"min_train_sessions": 5},
            {"holdout_end": "2030-01-01"},
            {"holdout_end": self.kwargs["holdout_start"]},
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_plan(self.calendar, **{**self.kwargs, **override})

    def test_pinned_artifact_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text("original")
            pinned = fingerprint_set([path])
            verify_fingerprints(pinned)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_fingerprints(pinned)


class ArchiveTests(unittest.TestCase):
    def test_unplanned_snapshot_insertion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            path = root / "snapshots/2023-01-01.json"
            path.write_text("{}")
            pipeline = Pipeline(argparse.Namespace(output_dir=root), {"selection": {"end": "2023-01-31"}})
            pipeline.round_results = [{"as_of": "2023-01-01"}]
            pipeline.completed_artifacts = fingerprint_set([path])
            pipeline.verify_archive("selection")
            (root / "snapshots/unplanned.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "archive differs"):
                pipeline.verify_archive("selection")

    def test_known_future_snapshot_cannot_enter_selection_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            (root / "snapshots/2023-02-01.json").write_text("{}")
            pipeline = Pipeline(argparse.Namespace(output_dir=root), {"selection": {"end": "2023-01-31"}})
            pipeline.round_results = [{"as_of": "2023-02-01"}]
            with self.assertRaisesRegex(ValueError, "beyond this phase"):
                pipeline.verify_archive("selection")


class PipelineIntegrationTests(unittest.TestCase):
    def prepare(self, root, periods=100):
        panel = fixture_panel(periods)
        panel.to_csv(root / "features.csv", index=False)
        (root / "candidates.json").write_text(json.dumps([{
            "candidate_id": "x", "op": "feature", "features": ["feature_x"], "hypothesis": "Synthetic fixed hypothesis",
        }]))
        return panel

    def arguments(self, root, panel, output="out", source="features.csv", initial=44, selection_end=69, holdout_end=89, interval=14):
        dates = sorted(panel["date"].unique())
        date = lambda index: str(pd.Timestamp(dates[index]).date())
        return [
            "--input", root / source, "--candidate-file", root / "candidates.json", "--output-dir", root / output,
            "--initial-as-of", date(initial), "--train-start", date(0),
            "--selection-start", date(initial + 1), "--selection-end", date(selection_end),
            "--holdout-start", date(selection_end + 1), "--holdout-end", date(holdout_end),
            "--admission-sessions", "20", "--min-train-sessions", "20", "--maintenance-days", str(interval),
            "--configs", ".8,0,2,2;.6,0,2,2",
        ]

    def test_dry_run_validates_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root)
            result = run_cli("run_pipeline.py", *self.arguments(root, panel), "--dry-run")
            self.assertFalse((root / "out").exists())
            self.assertEqual(len(json.loads(result.stdout)["rounds"]), 5)

    def test_end_to_end_freeze_precedes_holdout_and_future_mutation_cannot_reselect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root)
            args = self.arguments(root, panel)
            run_cli("run_pipeline.py", *args)
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertTrue(any(path.endswith("/alphax/factor_dsl.py") for path in manifest["fingerprints"]))
            selection_path = root / "out/trader/selection.json"
            selection_text = selection_path.read_text()
            frozen = json.loads(selection_text)
            self.assertEqual(frozen["status"], "selected")  # nontrivial successful admission and Trader gating
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertGreater(summary["holdout_invested_days"], 0)
            self.assertEqual(summary["selected_config"], frozen["selected_config"])
            events = [json.loads(line) for line in (root / "out/events.jsonl").read_text().splitlines()]
            frozen_index = next(i for i, event in enumerate(events) if event["event"] == "trader_frozen")
            for i, event in enumerate(events):
                if event["event"] == "step_started" and event["name"].startswith("miner_"):
                    cutoff = event["name"][len("miner_"):]
                    if cutoff > frozen["frozen_as_of"]:
                        self.assertGreater(i, frozen_index)
            for number, row in enumerate(summary["rounds"], 1):
                registry = json.loads((root / "out/miner" / row["as_of"] / "registry.json").read_text())
                self.assertEqual(registry["as_of"], row["as_of"])
                self.assertEqual(len(registry["run_history"]), number)
            changed = panel.copy()
            # Change holdout outcomes including the last selection-date label
            # whose execution lies in holdout. Selection must stay byte-identical.
            for column in ("target_open_close", "target_1d", "target_5d", "target_open_close_5d"):
                changed.loc[changed["date"] >= frozen["frozen_as_of"], column] *= -2
            changed.to_csv(root / "changed.csv", index=False)
            run_cli("run_pipeline.py", *self.arguments(root, panel, output="changed", source="changed.csv"))
            self.assertEqual((root / "changed/trader/selection.json").read_text(), selection_text)
            changed_summary = json.loads((root / "changed/summary.json").read_text())
            self.assertNotEqual(changed_summary["holdout"]["cumulative_return"], summary["holdout"]["cumulative_return"])
            changed_manifest = json.loads((root / "changed/manifest.json").read_text())
            self.assertNotEqual(changed_manifest["experiment_fingerprint"], manifest["experiment_fingerprint"])
            # Retrying a completed output must not mutate any archived decision.
            original_summary = (root / "out/summary.json").read_text()
            result = run_cli("run_pipeline.py", *args, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("never overwritten", result.stderr)
            self.assertEqual(selection_path.read_text(), selection_text)
            self.assertEqual((root / "out/summary.json").read_text(), original_summary)

    def test_same_experiment_has_distinct_runs_and_portable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root, periods=60)
            window = {"initial": 39, "selection_end": 49, "holdout_end": 59, "interval": 90}
            run_cli("run_pipeline.py", *self.arguments(root, panel, **window))
            shutil.copyfile(root / "features.csv", root / "copy.csv")
            rerun = self.arguments(root, panel, output="rerun", source="copy.csv", **window)
            run_cli("run_pipeline.py", *rerun, "--configs", "0.80, 0.0,2,2;0.60,0,2,2")

            original = root / "out"
            archived = root / "archive" / "moved"
            archived.parent.mkdir()
            original.rename(archived)
            first = json.loads((archived / "manifest.json").read_text())
            second = json.loads((root / "rerun/manifest.json").read_text())
            self.assertEqual(first["schema_version"], 1)
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["experiment_fingerprint"], second["experiment_fingerprint"])
            self.assertEqual(first["config"], second["config"])
            self.assertEqual(first["config"]["cost_rate"], .0001)
            for artifact in ("screener_holdout/predictions.csv", "trader/holdout_returns.csv"):
                self.assertEqual((archived / artifact).read_bytes(), (root / "rerun" / artifact).read_bytes())
            for key in ("selected_config", "holdout"):
                self.assertEqual(json.loads((archived / "summary.json").read_text())[key],
                                 json.loads((root / "rerun/summary.json").read_text())[key])

            runs = json.loads(run_cli("list_experiments.py", root).stdout)
            self.assertEqual({run["run_id"] for run in runs}, {first["run_id"], second["run_id"]})
            moved = next(run for run in runs if run["run_id"] == first["run_id"])
            self.assertEqual(moved["directory"], str(archived))
            self.assertEqual(moved["status"], "completed")
            self.assertEqual(moved["data_version"], first["inputs"]["panel"]["sha256"])
            self.assertEqual(moved["holdout"], first["holdout"])
            self.assertIsNotNone(moved["metrics"]["cumulative_return"])
            self.assertTrue({
                "trader/selection.json", "trader/holdout_returns.csv", "screener_holdout/predictions.csv",
            }.issubset(moved["artifacts"]))
            self.assertTrue(all(not Path(path).is_absolute() and (archived / path).is_file() for path in moved["artifacts"]))

    def test_empty_library_stays_cash_through_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root, periods=60)
            for column in ("target_open_close", "target_1d", "target_5d", "target_open_close_5d"):
                panel[column] = 0.
            panel.to_csv(root / "features.csv", index=False)
            run_cli("run_pipeline.py", *self.arguments(root, panel, initial=39, selection_end=49, holdout_end=59, interval=90))
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["status"], "no_feasible_strategy")
            self.assertEqual(summary["holdout"]["cumulative_return"], 0.)
            self.assertIsNone(summary["holdout"]["sharpe"])
            self.assertTrue(pd.read_csv(root / "out/trader/holdout_positions.csv").empty)

    def test_failed_stage_stops_before_holdout_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root)
            (root / "candidates.json").write_text("not valid JSON")
            result = run_cli("run_pipeline.py", *self.arguments(root, panel), check=False)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((root / "out/status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertTrue(status["stage"].startswith("miner_"))
            self.assertEqual(len(list((root / "out/logs").glob("*.log"))), 1)
            self.assertFalse((root / "out/trader/selection.json").exists())
            self.assertFalse((root / "out/summary.json").exists())
            self.assertFalse((root / "out/screener_holdout").exists())
            run, = json.loads(run_cli("list_experiments.py", root / "out").stdout)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["failed_stage"], status["stage"])
            self.assertIn(f"logs/{status['stage']}.log", run["artifacts"])
            self.assertIsNone(run["metrics"]["cumulative_return"])

    def run_with_hook(self, script, args):
        return subprocess.run(
            [sys.executable, "-c", script, *map(str, args)],
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30,
        )

    def test_resume_skips_committed_round_and_recomputes_partial_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root)
            args = self.arguments(root, panel)
            out = root / "out"
            stopped = self.run_with_hook("""
import os
from run_pipeline import Pipeline, main
original = Pipeline.mine
def stop_after_commit(self, round_plan):
    original(self, round_plan)
    os._exit(73)
Pipeline.mine = stop_after_commit
main()
""", args)
            self.assertEqual(stopped.returncode, 73, stopped.stderr)
            plan = json.loads((out / "plan.json").read_text())
            first, second = [row["as_of"] for row in plan["rounds"][:2]]
            first_checkpoint = (out / f"checkpoints/{first}.json").read_bytes()
            manifest = (out / "manifest.json").read_bytes()
            snapshot = out / f"snapshots/{first}.json"
            before_snapshot = snapshot.read_bytes()

            wrong_config = run_cli("run_pipeline.py", *args, "--resume", "--max-temporal", 1, check=False)
            self.assertIn("configuration", wrong_config.stderr)
            input_file = root / "features.csv"
            saved_input = input_file.read_bytes()
            input_file.write_bytes(saved_input + b"\n")
            wrong_input = run_cli("run_pipeline.py", *args, "--resume", check=False)
            self.assertIn("inputs", wrong_input.stderr)
            input_file.write_bytes(saved_input)
            snapshot.write_bytes(before_snapshot + b"\n")
            changed = run_cli("run_pipeline.py", *args, "--resume", check=False)
            self.assertIn("Committed Miner snapshot or registry changed", changed.stderr)
            snapshot.write_bytes(before_snapshot)
            self.assertFalse((out / "attempts").exists())

            stopped = self.run_with_hook("""
import os
import run_pipeline
original = run_pipeline.publish_json
def stop_before_checkpoint(path, payload):
    if path.parent.name == "checkpoints" and "round" in payload:
        os._exit(74)
    return original(path, payload)
run_pipeline.publish_json = stop_before_checkpoint
run_pipeline.main()
""", [*args, "--resume"])
            self.assertEqual(stopped.returncode, 74, stopped.stderr)
            self.assertFalse((out / f"checkpoints/{second}.json").exists())
            self.assertTrue((out / f"snapshots/{second}.json").exists())
            self.assertTrue((out / f".staging/{second}").is_dir())

            run_cli("run_pipeline.py", *args, "--resume")
            run_cli("run_pipeline.py", *self.arguments(root, panel, output="continuous"))
            self.assertEqual((out / f"checkpoints/{first}.json").read_bytes(), first_checkpoint)
            self.assertEqual((out / "manifest.json").read_bytes(), manifest)
            self.assertEqual(snapshot.read_bytes(), before_snapshot)
            self.assertEqual(len(list((out / "attempts").glob(f"*/snapshots/{second}.json"))), 1)
            self.assertEqual(len(list((out / "logs").glob(f"resume-*/miner_{second}.log"))), 2)
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(item.get("name") == f"miner_{first}" and item["event"] == "step_started"
                                 for item in events), 1)
            self.assertTrue(all(item["completed_rounds"] == [first] for item in events if item["event"] == "resumed"))
            continuous = root / "continuous"
            for row in plan["rounds"]:
                relative = f"snapshots/{row['as_of']}.json"
                self.assertEqual((out / relative).read_bytes(), (continuous / relative).read_bytes())
            for relative in ("rounds.csv", "trader/selection.json", "trader/holdout_returns.csv", "summary.json"):
                self.assertEqual((out / relative).read_bytes(), (continuous / relative).read_bytes())

    def test_resume_is_rejected_at_screener_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root, periods=60)
            args = self.arguments(root, panel, initial=39, selection_end=49, holdout_end=59, interval=90)
            stopped = self.run_with_hook("""
import os
from run_pipeline import Pipeline, main
Pipeline.screen = lambda self, phase: os._exit(75)
main()
""", args)
            self.assertEqual(stopped.returncode, 75, stopped.stderr)
            out = root / "out"
            self.assertTrue((out / "checkpoints/selection_closed.json").is_file())
            self.assertFalse((out / "screener_selection").exists())
            resumed = run_cli("run_pipeline.py", *args, "--resume", check=False)
            self.assertIn("before the selection Screener starts", resumed.stderr)
            self.assertFalse((out / "attempts").exists())

    def test_run_directory_allows_only_one_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = self.prepare(root, periods=60)
            args = self.arguments(root, panel, initial=39, selection_end=49, holdout_end=59, interval=90)
            script = """
import sys
from run_pipeline import Pipeline, main
def pause(self, pinned_inputs, resume):
    print("locked", flush=True)
    sys.stdin.readline()
    sys.exit(0)
Pipeline.run_locked = pause
main()
"""
            holder = subprocess.Popen(
                [sys.executable, "-c", script, *map(str, args)],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "locked")
                contender = run_cli("run_pipeline.py", *args, check=False)
                self.assertIn("Another process is already writing", contender.stderr)
                self.assertFalse((root / "out/manifest.json").exists())
            finally:
                holder.communicate(input="\n", timeout=10)


class TraderMaturityTests(unittest.TestCase):
    def test_delayed_stock_label_is_not_next_session_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({
                "date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-02", "2024-01-03", "2024-01-04"]),
                "ticker": ["A", "A", "B", "B", "B"],
                "target_open_close": [.2] * 5,
            })
            frame.to_csv(root / "features.csv", index=False)
            prediction = frame.rename(columns={"date": "datetime", "ticker": "instrument"})[["datetime", "instrument"]].assign(prediction=1.)
            prediction.loc[prediction["instrument"] == "A", "prediction"] = 2.
            prediction.to_csv(root / "predictions.csv", index=False)
            loaded = load_inputs(root / "predictions.csv", root / "features.csv", as_of="2024-01-03")
            self.assertEqual(loaded["date"].max(), pd.Timestamp("2024-01-03"))
            a = loaded[(loaded["date"] == "2024-01-02") & (loaded["ticker"] == "A")]
            self.assertTrue(a["target_open_close"].isna().all())
            with self.assertRaisesRegex(ValueError, "execution return"):
                simulate(loaded, TraderConfig(.8, 1., 1, 0), "2024-01-02", "2024-01-03", .0001)


if __name__ == "__main__":
    unittest.main()
