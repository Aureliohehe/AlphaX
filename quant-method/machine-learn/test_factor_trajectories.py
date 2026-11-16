"""Fixed trajectory budget, safe mutation schema, and PIT-free feature feedback."""
from __future__ import annotations

import unittest

import pandas as pd

from run_factor_trajectories import DEFAULT_MUTATIONS, EVAL_BUDGET, build_trajectories, feature_redundancy


class TrajectoryTests(unittest.TestCase):
    def test_bounded_lineage_and_no_untrusted_expression(self):
        rows = build_trajectories(DEFAULT_MUTATIONS)
        self.assertEqual(len(rows), EVAL_BUDGET)
        self.assertEqual([r["phase"] for r in rows].count("mutation"), 4)
        self.assertEqual(rows[-1]["parent_ids"], ["roc30_slope30_w20", "corr20_mean20_w10"])
        self.assertEqual(rows[3]["expression"], "Slope(Ref($close, 30)/$close, 20)")
        with self.assertRaisesRegex(ValueError, "Exactly four"):
            build_trajectories(DEFAULT_MUTATIONS[:3])
        for candidate in (
            {**DEFAULT_MUTATIONS[0], "window": 30},  # duplicates the parent
            {**DEFAULT_MUTATIONS[0], "window": "__import__('os')"},
            {**DEFAULT_MUTATIONS[0], "expression": "malicious"},
            {**DEFAULT_MUTATIONS[0], "parent_id": "unknown"},
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                build_trajectories([candidate] + DEFAULT_MUTATIONS[1:])

    def test_correlations_only_use_validation_same_day_features(self):
        index = pd.MultiIndex.from_product([['A', 'B', 'C'], pd.to_datetime(['2021-06-01', '2022-06-01'])],
                                           names=['instrument', 'datetime'])
        frame = pd.DataFrame({'child': [1, 100, 2, 0, 3, -100],
                              'parent': [3, 0, 2, 0, 1, 0]}, index=index)
        feedback = feature_redundancy(frame, 'child', ['parent'])
        self.assertAlmostEqual(feedback['parent'], -1.0)


if __name__ == '__main__':
    unittest.main()
