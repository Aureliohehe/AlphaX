"""Compatibility checks for reading old and unfinished pipeline records."""

import json
from pathlib import Path
import tempfile
import unittest

from alphax.experiment import CSI300_SCOPE, list_runs


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
            self.assertEqual(completed["experiment_type"], "pipeline")
            self.assertEqual(completed["data_version"], "legacy-panel-digest")
            self.assertEqual(completed["holdout"], window)
            self.assertEqual(completed["metrics"]["cumulative_return"], 0.1)
            self.assertIsNone(completed["run_id"])
            self.assertIsNone(completed["schema_version"])
            self.assertEqual(partial["status"], "unknown")
            self.assertIsNone(partial["metrics"]["cumulative_return"])
            self.assertEqual((legacy / "manifest.json").read_text(), manifest)

    def test_legacy_csi300_uses_research_metrics_and_original_data_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({
                "scope": CSI300_SCOPE,
                "holdout": ["2022-01-01", "2025-12-31"],
                "files": {f"/original/{name}": {"sha256": name}
                          for name in ("daily_pv.h5", "qlib_csi300.txt", "qlib_day.txt")},
            }))
            (root / "status.json").write_text('{"status":"completed_research"}')
            (root / "summary.json").write_text(
                '{"candidate_count":80,"strict_accepted":[],"holdout_evaluated":false}'
            )
            result, = list_runs(root)
            self.assertEqual(result["experiment_type"], "csi300_factor_research")
            self.assertEqual(result["status"], "completed")
            self.assertIsNone(result["run_id"])
            self.assertTrue(result["data_version"])
            self.assertEqual(result["holdout"], {"start": "2022-01-01", "end": "2025-12-31"})
            self.assertEqual(result["metrics"], {
                "candidate_count": 80, "strict_accepted_count": 0, "holdout_evaluated": False,
            })
