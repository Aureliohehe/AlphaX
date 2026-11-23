"""Offline record and research-flow checks using the real Miner CLI."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_csi300_research import run
from .test_temporal import run_cli


def fixture_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2018-09-03", periods=70).append([
        pd.bdate_range("2021-09-06", periods=70), pd.bdate_range("2025-09-01", periods=70),
    ])
    rng = np.random.default_rng(24)
    rows = []
    for index, day in enumerate(dates):
        for asset in range(16):
            score = asset - 7.5
            rows.append({
                "date": day, "ticker": f"S{asset:02d}", "feature_x": score,
                "return_1d": rng.normal(0, .01),
                "target_1d": .003 * score + rng.normal(0, .01) if index + 1 < len(dates) else np.nan,
                "target_5d": .005 * score + rng.normal(0, .01) if index + 5 < len(dates) else np.nan,
                "label_end_1d": dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                "label_end_5d": dates[index + 5] if index + 5 < len(dates) else pd.NaT,
            })
    return pd.DataFrame(rows)


class CSI300ExperimentTests(unittest.TestCase):
    def setUp(self):
        store = tempfile.TemporaryDirectory()
        self.addCleanup(store.cleanup)
        self.root = Path(store.name)
        self.panel = fixture_panel()
        self.source = self.root / "market"
        self.source.mkdir()
        (self.source / "daily_pv.h5").write_bytes(b"test fixture; the panel adapter is mocked")
        (self.source / "qlib_csi300.txt").write_text("fixture membership\n")
        days = self.panel["date"].drop_duplicates().dt.strftime("%Y-%m-%d")
        (self.source / "qlib_day.txt").write_text("\n".join(days) + "\n")
        self.prior = self.root / "candidates.json"
        self.prior.write_text("[]\n")
        self.schema = self.root / "schema.csv"
        self.panel.head(0).to_csv(self.schema, index=False)

    def write_panel(self, prices, membership, calendar, output, columns, start, end):
        self.panel[columns].to_csv(output, index=False)
        return {"member_rows": len(self.panel)}

    def run_fixture(self, out):
        with patch("run_csi300_research.build_panel", side_effect=self.write_panel):
            return run(out, self.source, self.prior, self.schema)

    def test_completed_run_and_zero_admission_are_cataloged_after_move(self):
        original = self.root / "original"
        result = self.run_fixture(original)
        self.assertEqual(result["strict_accepted"], ["csi_raw_feature_x"])
        self.assertTrue(result["holdout_evaluated"])
        frozen = json.loads((original / "holdout_selection_frozen.json").read_text())
        self.assertEqual(frozen, {"candidate_ids": ["csi_raw_feature_x"], "directions": {"csi_raw_feature_x": 1}})
        manifest = json.loads((original / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(set(manifest["inputs"]), {"prices", "membership", "calendar", "candidate_file", "feature_schema"})

        archived = self.root / "archive"
        original.rename(archived)
        self.panel[["target_1d", "target_5d"]] *= 0.0
        # Distinguish this input from the first in the source record as well.
        (self.source / "daily_pv.h5").write_bytes(b"fixture without a predictive signal")
        zero = self.root / "zero"
        result = self.run_fixture(zero)
        self.assertEqual(result["strict_accepted"], [])
        self.assertFalse(result["holdout_evaluated"])
        self.assertFalse((zero / "holdout.log").exists())
        self.assertEqual(json.loads((zero / "holdout_selection_frozen.json").read_text())["candidate_ids"], [])

        first, second = json.loads(run_cli("list_experiments.py", self.root).stdout)
        self.assertEqual(first["directory"], str(archived))
        self.assertEqual(first["experiment_type"], "csi300_factor_research")
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        self.assertEqual(first["run_id"], manifest["run_id"])
        self.assertEqual(first["metrics"], {"candidate_count": 1, "strict_accepted_count": 1, "holdout_evaluated": True})
        self.assertEqual(second["metrics"], {"candidate_count": 1, "strict_accepted_count": 0, "holdout_evaluated": False})
        self.assertNotEqual(first["data_version"], second["data_version"])
        self.assertNotEqual(first["experiment_fingerprint"], second["experiment_fingerprint"])
        self.assertTrue({"features.csv.gz", "candidates_frozen.json", "holdout_selection_frozen.json", "holdout.log"}
                        .issubset(first["artifacts"]))
        self.assertTrue(all(not Path(path).is_absolute() and (archived / path).is_file() for path in first["artifacts"]))

    def test_failed_feature_build_preserves_identity_and_partial_artifact_list(self):
        out = self.root / "failed"
        with patch("run_csi300_research.build_panel", side_effect=ValueError("invalid HDF fixture")):
            with self.assertRaisesRegex(ValueError, "invalid HDF fixture"):
                run(out, self.source, self.prior, self.schema)
        record, = json.loads(run_cli("list_experiments.py", out).stdout)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_stage"], "building_features")
        self.assertIn("invalid HDF fixture", record["error"])
        self.assertTrue(record["run_id"])
        self.assertTrue(record["experiment_fingerprint"])
        self.assertIn("candidates_frozen.json", record["artifacts"])
        self.assertNotIn("features.csv.gz", record["artifacts"])
        self.assertIsNone(record["metrics"]["holdout_evaluated"])

    def test_unreadable_input_is_still_visible_as_failed(self):
        out = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            run(out, self.source, self.prior, self.root / "missing.csv")
        record, = json.loads(run_cli("list_experiments.py", out).stdout)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_stage"], "preparing")
        self.assertTrue(record["run_id"])
        self.assertIsNone(record["experiment_fingerprint"])
