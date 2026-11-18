import json
from pathlib import Path
import tempfile
import unittest

from run_medium_frequency import miner_candidate_source


class MinerFrequencySourceTests(unittest.TestCase):
    def test_accepts_only_miner_sources_and_cutoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "candidates.json").write_text('[{"candidate_id": "only_miner"}]')
            registry = root / "registry.json"
            for source in ("external_json", "template_v1", "llm_langgraph:test:3"):
                values = {"registry_version": "miner-lite-v3", "as_of": "2021-12-31",
                          "generator": source, "candidate_count": 1}
                registry.write_text(json.dumps(values))
                _, _, generator, entries = miner_candidate_source(root, "2021-12-31")
                self.assertEqual(generator, source)
                self.assertEqual([row["candidate_id"] for row in entries], ["only_miner"])
            for update in ({"generator": "builtin"}, {"as_of": "2022-01-01"}, {"candidate_count": 2}):
                registry.write_text(json.dumps({**values, **update}))
                with self.assertRaises(ValueError):
                    miner_candidate_source(root, "2021-12-31")
