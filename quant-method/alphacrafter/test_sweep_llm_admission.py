"""Frozen LLM gate sweep is diagnostic only; strict rules remain untouched."""
import unittest

from sweep_llm_admission import build_all_admit_ladder, gate, retrospective_library, run_sweep


class AdmissionSweepTests(unittest.TestCase):
    def fixture(self):
        metric = {"mean_daily_ic": .001, "mean_daily_rank_ic": .001,
                  "icir": .01, "rank_icir": .01, "rank_ic_hit_ratio": .52,
                  "coverage": .95, "turnover": .03}
        row = {"candidate_id": "x", "origin": "frozen_llm", "direction_from_train": 1,
               "strict_accepted": False, "horizons": {"1": {"train": metric, "validation": metric},
                                             "5": {"train": {**metric, "rank_ic_hit_ratio": .49},
                                                   "validation": {**metric, "rank_ic_hit_ratio": .49}}}}
        return row

    def test_one_day_only_and_near_coinflip_multihorizon_are_explicitly_distinct(self):
        row = self.fixture()
        by_name = {r["policy"]: r for r in run_sweep([row])}
        self.assertEqual(by_name["positive_1d_hit_above_50"]["admitted_ids"], ["x"])
        self.assertEqual(by_name["positive_1d_5d_hit_above_50"]["admitted_ids"], [])
        self.assertEqual(by_name["positive_1d_5d_hit_above_49"]["admitted_ids"], [])  # strict >
        self.assertEqual(by_name["paper_1d_5d"]["admitted_ids"], [])
        row["horizons"]["5"]["validation"]["rank_ic_hit_ratio"] = .495
        row["horizons"]["5"]["train"]["rank_ic_hit_ratio"] = .495
        self.assertTrue(gate(row, (1, 5), 0, .49))
        row["horizons"]["5"]["validation"]["rank_icir"] = float("nan")
        self.assertFalse(gate(row, (1, 5), 0, .49))

    def test_ladder_reaches_every_factor_without_silently_weakening_coverage(self):
        weak = self.fixture()
        weak["candidate_id"] = "weak"
        weak["horizons"]["5"]["train"]["icir"] = -.1
        strong = self.fixture()
        strong["candidate_id"] = "strong"
        rows = [strong, weak]
        ladder = build_all_admit_ladder(rows, steps=10)
        self.assertEqual(ladder["steps"][0]["admitted_ids"], [])
        self.assertEqual(ladder["steps"][-1]["admitted_ids"], ["strong", "weak"])
        self.assertLess(ladder["all_admit_floors"]["5"]["icir"], -.1)
        self.assertLessEqual(ladder["first_entry"]["strong"]["step"], ladder["first_entry"]["weak"]["step"])
        self.assertEqual(ladder["coverage_floor"], .9)
        weak["horizons"]["1"]["train"]["coverage"] = .9
        with self.assertRaisesRegex(ValueError, "coverage and turnover"):
            build_all_admit_ladder(rows, 10)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_all_admit_ladder([strong, strong], 10)

    def test_research_library_is_not_backdated_or_strict(self):
        row = self.fixture()
        policy = {"policy": "research", "admitted_ids": ["x"]}
        result = retrospective_library([row], {"x": {"candidate_id": "x", "expression": {"op": "feature", "feature": "alpha158_roc10"}, "hypothesis": "test"}}, policy, "2022-12-30")
        self.assertEqual(result["library_scope"], "retrospective_diagnostic")
        self.assertFalse(result["factors"][0]["strict_paper_accepted"])
        self.assertNotIn("as_of", result)
        self.assertNotIn("created_at", result["factors"][0])
        self.assertEqual(result["factors"][0]["historical_metric_cutoff"], "2022-12-30")


if __name__ == "__main__":
    unittest.main()
