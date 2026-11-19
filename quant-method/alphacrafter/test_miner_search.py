import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from miner_lite import Candidate
from miner_search import changed_fields, rolling_feedback, run_search, stable_horizons, validation_tasks


class SearchTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        self.frame = pd.DataFrame({"date": dates.repeat(4), "ticker": list("ABCD") * 40,
                                   "x": [1., 2., 3., 4.] * 40})
        self.targets = {h: pd.Series(np.tile([-.02, -.01, .01, .02], 40)) for h in (1, 5)}
        self.ends = {h: self.frame.date.map(dict(zip(dates[:-h], dates[h:]))) for h in (1, 5)}
        self.candidate = Candidate("x", "feature", ("x",), "test")

    def test_fold_labels_are_purged_and_direction_uses_prefix(self):
        first = rolling_feedback(self.frame, self.candidate, ["x"], self.targets, self.ends)
        changed = {h: values.copy() for h, values in self.targets.items()}
        # Corrupt only labels maturing after the first evaluation block.
        end = pd.Timestamp(first["folds"][0]["end"])
        for h in changed:
            changed[h].loc[self.ends[h] > end] *= -100
        second = rolling_feedback(self.frame, self.candidate, ["x"], changed, self.ends)
        self.assertEqual(first["folds"][0]["direction"], second["folds"][0]["direction"])
        for index in (0, 1):
            self.assertEqual(first["folds"][index]["metrics"]["rows"], second["folds"][index]["metrics"]["rows"])
            self.assertEqual(first["folds"][index]["metrics"]["mean_daily_rank_ic"], second["folds"][index]["metrics"]["mean_daily_rank_ic"])
        five = first["folds"][1]
        self.assertEqual(five["metrics"]["rows"], 20)

    def test_budget_duplicates_and_feedback(self):
        calls = []
        def propose(count, feedback):
            calls.append((count, json.loads(feedback)))
            return [self.candidate] * count
        candidates, audit = run_search(propose, self.frame, ["x"], self.targets, self.ends,
                                       rounds=5, budget=3, batch_size=2)
        self.assertEqual([c[0] for c in calls], [2, 1])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(audit["budget_spent"], 3)
        self.assertEqual(calls[0][1], [])
        self.assertEqual(calls[1][1][0]["expression"], self.candidate.expression)
        self.assertEqual(calls[1][1][0]["next_validation_tasks"][0]["control_id"], "x")
        json.dumps(audit, allow_nan=False)

    def test_diagnostics_assign_horizon_specific_followups(self):
        tasks = validation_tasks({
            "candidate_id": "control", "direction_changes": 0,
            "folds": [
                {"horizon": 1, "fold": 1, "failure_reasons": ["high_or_unknown_turnover"]},
                {"horizon": 5, "fold": 1, "failure_reasons": ["low_coverage"]},
                {"horizon": 5, "fold": 2, "failure_reasons": []},
            ],
        })
        self.assertEqual([(t["horizon"], t["primary_metric"]) for t in tasks],
                         [(1, "turnover"), (5, "coverage")])
        self.assertEqual(tasks[1]["failing_folds"], [1])
        self.assertEqual(tasks[1]["control_id"], "control")

    def test_followup_rejects_two_edits_and_limits_the_lineage(self):
        from miner_lite import candidate_from_dict

        def candidate(cid, feature, window, parent=None):
            return candidate_from_dict({
                "candidate_id": cid, "hypothesis": "fixture", "control_id": parent,
                "expression": {"op": "ts_mean", "feature": feature, "window": window},
            })

        root = candidate("root", "x5", 5)
        two_edits = candidate("two_edits", "x10", 10, "root")
        followup = candidate("followup", "x5", 10, "root")
        retry = candidate("retry", "x5", 20, "followup")
        disguised = candidate("disguised", "x10", 20)
        fresh = candidate("fresh", "y", 20)
        proposals = iter([[root], [two_edits, followup], [retry, disguised, fresh]])
        frame = self.frame.assign(x5=self.frame.x, x10=self.frame.x, y=self.frame.x)
        results, audit = run_search(lambda count, feedback: next(proposals), frame,
                                    ["x5", "x10", "y"], self.targets, self.ends,
                                    rounds=3, batch_size=3, budget=9)
        self.assertEqual([c.candidate_id for c in results], ["root", "followup", "fresh"])
        self.assertEqual(audit["rounds"][1]["rejected"][0]["ast_edits"], 2)
        self.assertEqual([r["reason"] for r in audit["rounds"][2]["rejected"]],
                         ["lineage_follow_up_limit_reached", "related_hypothesis_requires_control"])
        self.assertIsNone(changed_fields({"op": "feature", "feature": "x"},
                                         {"op": "rank", "input": {"op": "feature", "feature": "x"}}))

    def test_reserve_replaces_missing_slots_with_new_hypotheses(self):
        calls = []
        def propose(count, feedback):
            calls.append((count, json.loads(feedback)[-1]["round_policy"]))
            return [self.candidate]
        frame = self.frame.assign(y=self.frame.x)
        candidates, audit = run_search(propose, frame, ["x", "y"], self.targets, self.ends,
                                       rounds=2, budget=6, batch_size=2, reserve=2)
        self.assertEqual([c.features for c in candidates], [("x",), ("y",)])
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit["rounds"][-1]["followup_limit"], 0)
        self.assertTrue(audit["rounds"][-1]["supplemental"])
        self.assertEqual(audit["budget_spent"], 6)

    def test_stability_requires_majority_with_same_direction(self):
        folds = [{"horizon": 1, "direction": direction,
                  "metrics": {"mean_daily_rank_ic": rank_ic}}
                 for direction, rank_ic in [(1, .01), (-1, -.02), (1, -.01)]]
        self.assertEqual(stable_horizons({"folds": folds}), [])
        folds[-1]["metrics"]["mean_daily_rank_ic"] = .02
        self.assertEqual(stable_horizons({"folds": folds}), [1])

    def test_seeded_llm_keeps_controls_and_rejects_unrelated_components(self):
        from miner_lite import candidate_from_dict, seeded_llm_candidates

        base = Candidate("vol60", "feature", ("vol60",), "control")
        valid = candidate_from_dict({
            "candidate_id": "vol_smooth", "control_id": "vol60",
            "expression": {"op": "ts_mean", "feature": "vol60", "window": 10},
        })
        invalid = Candidate("momentum", "feature", ("momentum",), "wrong family",
                            control_id="vol60")
        zscore = candidate_from_dict({
            "candidate_id": "vol_zscore", "control_id": "vol60",
            "expression": {"op": "ts_zscore", "feature": "vol60", "window": 10},
        })
        audit = []
        self.assertEqual(seeded_llm_candidates([base], [invalid, zscore, valid], audit), [base, valid])
        self.assertEqual([row["status"] for row in audit], ["rejected", "rejected", "evaluated"])

    def test_insufficient_sessions_fail(self):
        with self.assertRaises(ValueError):
            rolling_feedback(self.frame.iloc[:8], self.candidate, ["x"], self.targets, self.ends)

    def test_llm_prompt_preserves_slots_and_requests_testable_hypotheses(self):
        from miner_lite import generate_llm_candidates

        response = json.dumps([{"candidate_id": "tail", "hypothesis": "test",
                                "expression": {"op": "ts_quantile", "feature": "x",
                                               "window": 20, "quantile": .25}}])
        with patch("miner_lite._deepseek_completion", return_value=response) as provider:
            results = generate_llm_candidates("deepseek-chat", ["x"], "weekly context", 2)
        self.assertEqual([candidate.candidate_id for candidate in results], ["tail"])
        prompt = provider.call_args.args[0][1]["content"]
        self.assertIn("holding horizon", prompt)
        self.assertIn("one observable result that would count against", prompt)
        self.assertIn("Generate exactly 2", prompt)
        self.assertIn("weekly context", prompt)

    def test_langgraph_specialists_share_requested_budget(self):
        from langgraph_miner import _fan_out
        with patch("langgraph_miner.Send", side_effect=lambda node, state: state):
            tasks = _fan_out({"available": ["x"], "model": "mock", "market_context": "context",
                              "requested_count": 10, "agent_count": 3})
        self.assertEqual([task["per_agent_count"] for task in tasks], [4, 3, 3])

    def test_cli_passes_only_training_rows_to_search(self):
        from miner_lite import main
        from test_temporal import fixture_panel
        from miner_search import run_search as real_search
        panel = fixture_panel(100)
        sessions = sorted(panel.date.unique())
        def day(index):
            return str(pd.Timestamp(sessions[index]).date())
        observed = []
        def inspect(propose, frame, available, targets, ends, **kwargs):
            self.assertLessEqual(frame.date.max(), pd.Timestamp(day(59)))
            for h, target in targets.items():
                self.assertTrue(target.loc[ends[h] > pd.Timestamp(day(59))].isna().all())
            result = real_search(propose, frame, available, targets, ends, **kwargs)
            observed.append(result[1])
            return result
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out"
            panel.to_csv(source, index=False)
            args = ["miner", "--input", str(source), "--output-dir", str(output),
                    "--train-start", day(0), "--train-end", day(59),
                    "--validation-start", day(60), "--validation-end", day(79),
                    "--backtest-start", day(80), "--backtest-end", day(99),
                    "--llm-model", "mock", "--search-rounds", "2", "--search-budget", "2",
                    "--search-batch-size", "1"]
            candidate = Candidate("x", "feature", ("feature_x",), "fixture")
            with patch("sys.argv", args), patch("miner_search.run_search", side_effect=inspect), patch(
                "langgraph_miner.generate_langgraph_candidates", return_value=[candidate]
            ):
                main()
            report = json.loads((output / "search_feedback.json").read_text())
            self.assertEqual(report, observed[0])
            registry = json.loads((output / "registry.json").read_text())
            self.assertEqual(registry["selection"]["paper_min_hit_deviation"], .1)


if __name__ == "__main__":
    unittest.main()
