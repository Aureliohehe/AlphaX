"""Fixed-budget exploratory batch safety and diagnostic tests."""
import tempfile
import unittest
from pathlib import Path

from run_candidate_batch import FAMILIES, canonical, failure_reasons, generate, remove_tradable_artifacts


class CandidateBatchTests(unittest.TestCase):
    def test_structural_dedup_budget_and_hypothesis(self):
        proposals = [
            {"candidate_id": "llm_a", "expression": {"op": "feature", "feature": "alpha158_roc10"},
             "hypothesis": "trend may persist"},
            {"candidate_id": "llm_dup", "features": ["alpha158_roc10"], "op": "feature",
             "hypothesis": "same factor another spelling"},
        ]
        available = {x for group in FAMILIES.values() for x in group}
        chosen, audit = generate(proposals, available, 3)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(audit[1]["duplicate_of"], "llm_a")
        self.assertEqual(audit[2]["duplicate_of"], "llm_a")
        self.assertEqual(chosen[1]["candidate_id"], "batch_raw_alpha158_roc30")
        self.assertEqual(chosen[0]["hypothesis"], "trend may persist")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            generate([{"candidate_id": "future", "op": "feature", "features": ["target_open_close"],
                       "hypothesis": "leak"}], available, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate ID"):
            generate([proposals[0], proposals[0]], available, 2)

    def test_retrospective_batch_cannot_export_backdated_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "snapshots").mkdir()
            (directory / "snapshots/2022-12-30.json").write_text("{}")
            for name in ("registry.json", "factor_library.json", "accepted_factors.json", "predictions.csv"):
                (directory / name).write_text("test")
            (directory / "candidate_metrics.json").write_text("[]")
            remove_tradable_artifacts(directory)
            self.assertEqual(sorted(x.name for x in directory.iterdir()),
                             ["RETROSPECTIVE_ONLY.txt", "candidate_metrics.json"])

    def test_canonical_commutativity_only_when_valid(self):
        a, b = {"op": "feature", "feature": "A"}, {"op": "feature", "feature": "B"}
        self.assertEqual(canonical({"op": "sum", "inputs": [a, b]}), canonical({"op": "sum", "inputs": [b, a]}))
        self.assertNotEqual(canonical({"op": "difference", "inputs": [a, b]}), canonical({"op": "difference", "inputs": [b, a]}))

    def test_gate_report_uses_direction_and_exact_boundaries(self):
        passing = {"mean_daily_ic": -.03, "mean_daily_rank_ic": -.03,
                   "icir": -.3, "rank_icir": -.3, "rank_ic_hit_ratio": .39,
                   "coverage": .99, "turnover": .2}
        self.assertEqual(failure_reasons(passing, -1, 1), [])
        bad = {**passing, "rank_ic_hit_ratio": .4, "icir": -.20, "coverage": .9, "turnover": .4}
        self.assertEqual(set(failure_reasons(bad, -1, 1)), {"icir", "hit_ratio", "coverage", "turnover"})
        self.assertIn("mean_daily_ic", failure_reasons({**passing, "mean_daily_ic": float("nan")}, -1, 1))


if __name__ == "__main__":
    unittest.main()
