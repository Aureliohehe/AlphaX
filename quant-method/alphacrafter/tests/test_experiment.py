"""Compatibility checks for reading old and unfinished pipeline records."""

import json
from pathlib import Path
import tempfile
import unittest

from alphax.experiment import list_runs


class ExperimentCatalogTests(unittest.TestCase):
    def test_legacy_runs_are_read_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy, unfinished, unrelated = (root / name for name in ("legacy", "unfinished", "other"))
            window = {"start": "2024-01-02", "end": "2024-06-30"}
            plan = json.dumps({"runner_version": "pit-pipeline-v1", "holdout": window})
            summary = json.dumps({"status": "selected", "holdout": {"cumulative_return": 0.1}})
            for directory in (legacy, unfinished):
                directory.mkdir()
                (directory / "plan.json").write_text(plan)
                (directory / "summary.json").write_text(summary)
            source = "/original/location/panel.csv"
            manifest = json.dumps({
                "parameters": {"input": source},
                "fingerprints": {source: {"sha256": "legacy-panel-digest"}},
            })
            (legacy / "manifest.json").write_text(manifest)
            (legacy / "status.json").write_text('{"status":"completed"}')
            unrelated.mkdir()
            (unrelated / "manifest.json").write_text('{"scope":"other research"}')

            completed, partial = list_runs(root)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["data_version"], "legacy-panel-digest")
            self.assertEqual(completed["holdout"], window)
            self.assertEqual(completed["metrics"]["cumulative_return"], 0.1)
            self.assertIsNone(completed["run_id"])
            self.assertIsNone(completed["schema_version"])
            self.assertEqual(partial["status"], "unknown")
            self.assertIsNone(partial["metrics"]["cumulative_return"])
            self.assertEqual((legacy / "manifest.json").read_text(), manifest)
