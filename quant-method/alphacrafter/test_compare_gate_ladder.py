"""All monotone ladder steps compare only unique factor sets, without selection leaks."""
import unittest

from compare_gate_ladder import distinct_sets, status_summary


class CompareGateLadderTests(unittest.TestCase):
    def test_distinct_sets_preserve_first_occurrence_and_full_endpoint(self):
        steps = [{"step": i, "fraction": i / 3, "admitted_ids": names, "floors": {}}
                 for i, names in enumerate(([], [], ["a"], ["a", "b"]))]
        result = distinct_sets(steps)
        self.assertEqual([r["first_step"] for r in result], [0, 2, 3])
        self.assertEqual(result[-1]["admitted_ids"], ["a", "b"])
        steps[3]["admitted_ids"] = ["b"]
        with self.assertRaisesRegex(ValueError, "Non-monotone"):
            distinct_sets(steps)
        steps[3]["admitted_ids"] = ["a", "a"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            distinct_sets(steps)

    def test_best_is_posthoc_only_and_ties_select_earlier(self):
        rows = [{"status": "simulated", "first_step": step, "admitted_ids": [str(step)],
                 "metrics": {"cumulative_return": net}}
                for step, net in ((1, .02), (5, .02), (10, -.1))]
        summary = status_summary(rows)
        self.assertEqual(summary["best_first_step"], 1)
        self.assertEqual(summary["best_variant"], "step_001")
        self.assertIn("post-hoc", summary["warning"])
        rows[-1]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "All comparison groups"):
            status_summary(rows)


if __name__ == "__main__":
    unittest.main()
