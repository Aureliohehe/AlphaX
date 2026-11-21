import json
from pathlib import Path
import tempfile
import unittest

from run_medium_frequency import gate_weekly_llm_trials, miner_candidate_source
from trader_harness import AUDIT_CHECKS


class MinerFrequencySourceTests(unittest.TestCase):
    def test_weekly_replacement_must_beat_core_at_both_costs(self):
        def trial(name, standard, double):
            execution = {
                "max_drawdown": -.08, "calendar_year_returns": {"2022": -.01, "2023": .20},
                "execution_audit": dict.fromkeys(AUDIT_CHECKS, True),
            }
            return {"horizon": 5, "strategy_id": name, "status": "feasible_in_proxy",
                    **execution, "cumulative_return": standard,
                    "double_cost": {**execution, "cumulative_return": double}}

        core = trial("core4", .10, .09)
        worse_under_stress = trial("core4_replace_stress", .12, .08)
        better = trial("core4_replace_better", .12, .11)
        worse_in_one_year = trial("core4_replace_year", .12, .11)
        worse_in_one_year["calendar_year_returns"] = {"2022": .15, "2023": .18}
        gate_weekly_llm_trials([worse_under_stress, better, worse_in_one_year, core])
        self.assertEqual(worse_under_stress["status"], "not_feasible")
        self.assertFalse(worse_under_stress["paired_execution"]["checks"]["2"]["higher_net_return"])
        self.assertEqual(better["status"], "feasible_in_proxy")
        self.assertFalse(better["paired_execution"]["replacement_admitted"])
        self.assertEqual(worse_in_one_year["status"], "not_feasible")
        annual = worse_in_one_year["paired_execution"]
        self.assertTrue(annual["checks"]["1"]["no_worse_worst_year"])
        self.assertFalse(annual["checks"]["1"]["no_worse_each_year"])
        self.assertAlmostEqual(annual["calendar_year_return_deltas"]["1"]["2023"], -.02)

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
