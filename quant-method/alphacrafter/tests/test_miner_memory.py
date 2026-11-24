from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from alphax.factor_dsl import Candidate
from alphax.miner_memory import MinerMemory, main as import_feedback, research_metadata
from miner_lite import main as run_miner
from temporal import TEMPORAL_POLICY_VERSION
from .test_temporal import fixture_panel


class MinerMemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "memory.sqlite"
        self.memory = MinerMemory(self.db)
        self.audit = {
            "as_of": "2020-03-01", "label_mode": "execution", "policy_version": TEMPORAL_POLICY_VERSION,
            "target_sources": {"1d": "target_open_close", "5d": "target_open_close_5d"},
            "label_end_source": {"1": "explicit_raw_label_dates", "5": "explicit_raw_label_dates"},
            "windows": {"train": ["2020-01-01", "2020-02-01"], "validation": ["2020-02-02", "2020-03-01"]},
        }
        self.metadata = research_metadata(self.audit, "us", "fixture", "source-v1")

    def record(self, metadata, available_at, **changes):
        row = {
            "candidate_id": "old", "ast": {"op": "feature", "feature": "feature_x"},
            "hypothesis": "Prior mean reversion test", "control_id": None, "details": {},
            "horizon": 1, "kind": "train", "evaluation_end": "2020-01-31",
            "evidence": {"failure_reasons": ["insufficient_rank_ic"], "direction": 1,
                         "metrics": {"mean_daily_rank_ic": .01, "rank_icir": .12}},
            **changes,
        }
        with patch("alphax.miner_memory.utc_now", return_value=available_at + "T00:00:00+00:00"):
            return self.memory.record(metadata, "fixture.json", [row])

    def test_recall_filters_availability_cutoff_scope_and_features(self):
        known = self.record(self.metadata, "2020-04-01")
        self.assertEqual(known, self.record(self.metadata, "2020-04-02"))
        self.record(self.metadata, "2020-04-01", kind="validation", evaluation_end="2020-02-20")
        self.record(self.metadata, "2020-05-01", horizon=5)
        self.record({**self.metadata, "market": "cn"}, "2020-04-01", horizon=5)
        wrong_target = {**self.metadata, "target_sources": {**self.metadata["target_sources"], "5d": "target_5d"}}
        self.record(wrong_target, "2020-04-01", horizon=5)
        self.record({**self.metadata, "evaluation_policy_version": "legacy"}, "2020-04-01", horizon=5)
        self.record(self.metadata, "2020-04-01", horizon=5, ast={"op": "feature", "feature": "missing"})
        current = {**self.metadata, "as_of": "2020-05-01"}
        results = self.memory.recall(current, ["feature_x"], {}, 6)
        self.assertEqual([(row["run_id"], row["kind"], row["horizon"]) for row in results], [(known, "train", 1)])
        self.assertEqual(results[0]["available_at"], "2020-04-01T00:00:00+00:00")
        self.assertEqual(self.memory.recall(current, ["feature_x"], {"other": {"op": "feature", "feature": "y"}}, 6), [])

    def test_import_legacy_report_twice(self):
        report = {
            "scope": "training_only_research_not_admission",
            "candidates": [{
                "candidate_id": "old", "expression": "feature_x", "hypothesis": "Test",
                "folds": [{"horizon": 1, "fold": 1, "start": "2020-01-15", "end": "2020-01-31",
                           "direction": 1, "failure_reasons": ["low_coverage"], "metrics": {}}],
            }],
        }
        source = self.root / "search_feedback.json"
        source.write_text(json.dumps(report))
        (self.root / "temporal_audit.json").write_text(json.dumps(self.audit))
        (self.root / "candidates.json").write_text(json.dumps([{
            "candidate_id": "old", "expression": {"op": "feature", "feature": "feature_x"}, "hypothesis": "Test",
        }]))
        cli = ["import", "--db", str(self.db), "--search-feedback", str(source),
               "--universe", "fixture", "--market", "us", "--data-version", "v1"]
        for day in ("2020-04-01", "2020-05-01"):
            with patch("sys.argv", cli), patch("alphax.miner_memory.utc_now", return_value=day + "T10:00:00+00:00"):
                import_feedback()
        recalled = self.memory.recall({**self.metadata, "as_of": "2020-04-02"}, ["feature_x"], {}, 6)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["failure_reasons"], ["low_coverage"])

    def test_cli_recall_reaches_prompt_and_writes_train_and_validation_only(self):
        source = self.root / "panel.csv"
        panel = fixture_panel(100)
        panel.to_csv(source, index=False)
        dates = sorted(panel.date.unique())

        def day(index):
            return pd.Timestamp(dates[index]).date().isoformat()

        past = {**self.metadata, "as_of": day(30),
                "label_end_source": {"1": "legacy_feature_row_inference", "5": "legacy_feature_row_inference"},
                "windows": {"train": [day(0), day(30)], "validation": [day(31), day(35)]}}
        self.record(past, day(35), evaluation_end=day(25))
        common = ["miner", "--input", str(source), "--train-start", day(0), "--train-end", day(59),
                  "--validation-start", day(60), "--validation-end", day(79),
                  "--backtest-start", day(80), "--backtest-end", day(99), "--llm-model", "mock"]
        memory_args = ["--memory-db", str(self.db), "--memory-universe", "fixture", "--research-market", "us"]
        candidate = Candidate("new", "feature", ("feature_x",), "New hypothesis")
        prompts = {}
        for name, options in (("off", []), ("on", memory_args)):
            args = common + ["--output-dir", str(self.root / name)] + options
            with patch("sys.argv", args), patch("langgraph_miner.generate_langgraph_candidates",
                                                 return_value=[candidate]) as model:
                run_miner()
            prompts[name] = model.call_args.args[2]

        audit = json.loads((self.root / "on" / "memory_recall.json").read_text())
        self.assertEqual(len(audit["evidence_ids"]), 1)
        evidence_id = audit["evidence_ids"][0]
        self.assertIn(evidence_id, prompts["on"])
        self.assertNotIn(evidence_id, prompts["off"])
        with closing(sqlite3.connect(self.db)) as connection:
            stored = connection.execute("SELECT DISTINCT kind FROM evaluations WHERE run_id = ?",
                                        (audit["published_run_id"],)).fetchall()
        self.assertCountEqual(stored, [("train",), ("validation",)])
