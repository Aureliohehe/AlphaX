"""The standalone example leaves inspectable artifacts and refuses overwrite."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .test_temporal import run_cli


class OfflineExampleTests(unittest.TestCase):
    def test_persistent_example_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "example"
            run_cli("run_offline_example.py", "--output-dir", output)
            self.assertEqual(json.loads((output / "pipeline/status.json").read_text())["status"], "completed")
            summary = json.loads((output / "pipeline/summary.json").read_text())
            self.assertGreater(summary["holdout_invested_days"], 0)
            self.assertIn("SYNTHETIC DATA", (output / "README.txt").read_text())
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob("*") if p.is_file()}
            result = run_cli("run_offline_example.py", "--output-dir", output, check=False)
            self.assertNotEqual(result.returncode, 0)
            after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in output.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
