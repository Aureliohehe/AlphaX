import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langgraph_miner import _fan_out, _generate_agent_candidates
from miner_lite import Candidate
from weekly_llm_feedback import load_weekly_feedback, load_xgboost_feedback


class WeeklyFeedbackTests(unittest.TestCase):
    def test_mechanism_first_rejects_invalid_claims_and_keeps_the_frozen_falsifier(self):
        mechanism = {"mechanism": "Stable volume can sustain reversals.",
                     "comparison": "Matched stocks with unstable volume.",
                     "expected_effect": "Higher five-session rank IC.",
                     "falsifier": "Nonpositive paired rank IC delta.", "holding_sessions": 5,
                     "control_id": "prior_factor"}
        state = {"role": "researcher", "model": "mock", "available": ["x"],
                 "market_context": "fixture", "per_agent_count": 2, "mechanism_first": True}

        def write_expression(model, available, context, count, **kwargs):
            self.assertEqual(count, 1)
            frozen = json.loads(context.split("Frozen mechanisms: ", 1)[1])
            identifier, = frozen
            self.assertEqual(frozen[identifier], mechanism)
            return [Candidate(identifier, "feature", ("x",), "Changed hypothesis"),
                    Candidate("unmatched", "feature", ("x",), "Unknown mechanism")]

        with patch("langgraph_miner.request_llm_json", return_value=json.dumps(
            [mechanism, {**mechanism, "falsifier": ""}]
        )), patch("langgraph_miner.generate_llm_candidates", side_effect=write_expression):
            result = _generate_agent_candidates(state)
        candidate, = result["candidate_payloads"]
        self.assertIn(mechanism["falsifier"], candidate["hypothesis"])
        self.assertEqual(candidate["control_id"], "prior_factor")
        self.assertNotIn("Changed hypothesis", candidate["hypothesis"])
        self.assertEqual([row["reason"] for row in result["audit"] if row["status"] == "mechanism_rejected"],
                         ["invalid_mechanism_schema", "expression_id_not_frozen"])

    def test_xgboost_feedback_keeps_cost_and_year_deltas_and_rejects_future_trials(self):
        trial = {"start": "2024-01-02", "end": "2025-12-31", "trading_days": 485,
                 "cumulative_return": .1, "max_drawdown": -.1, "mean_daily_turnover": .01,
                 "calendar_year_returns": {"2024": 0., "2025": .1}}
        changed = {**trial, "cumulative_return": .2, "calendar_year_returns": {"2024": .3, "2025": -.08}}
        row = {"control": "x", "id": "variant", "expression": "ts_mean(x, 10)", "execution_passes": True,
               "results": {"core": {"costs": {"1": trial, "2": trial}},
                           "augmented": {"costs": {"1": changed, "2": changed}}}}
        replay = {"window": ["2024-01-01", "2025-12-31"], "augmentation_weight": .25,
                  "rounds": {"first": {"miner_as_of": "2023-12-31", "variants": [row]}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay_summary.json"
            path.write_text(json.dumps(replay))
            with self.assertRaisesRegex(ValueError, "Miner cutoff"):
                load_xgboost_feedback(path, "2025-06-30", ["x"])
            result = load_xgboost_feedback(path, "2025-12-31", ["x", "y"])
            self.assertEqual(result["prior_expressions"], ["ts_mean(x, 10)"])
            self.assertEqual(result["by_control"]["y"], [])
            self.assertEqual(result["by_control"]["x"][0]["augmentation_weight"], .25)
            paired = result["by_control"]["x"][0]["paired_execution"]
            self.assertFalse(paired["checks"]["1"]["no_worse_each_year"])
            self.assertAlmostEqual(paired["calendar_year_return_deltas"]["2"]["2025"], -.18)
            changed["end"] = "2026-01-05"
            path.write_text(json.dumps(replay))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_xgboost_feedback(path, "2025-12-31", ["x"])
            changed["end"] = trial["end"]
            row["control"] = None
            path.write_text(json.dumps(replay))
            self.assertEqual(load_xgboost_feedback(path, "2025-12-31", None)["standalone"][0]["candidate_id"],
                             "variant")
            with self.assertRaisesRegex(ValueError, "proposal mode"):
                load_xgboost_feedback(path, "2025-12-31", ["x"])

    def test_imports_failed_checks_and_cost_deltas_with_a_temporal_cutoff(self):
        prior = {"cumulative_return": .06, "annualized_return": .03,
                 "sharpe": .33, "max_drawdown": -.11}
        stress = {**prior, "cumulative_return": .04, "sharpe": .20}
        files = {
            "status.json": {"status": "completed_research"},
            "strategy_selection.json": {"status": "selected_in_iteration_proxy",
                                        "selection_window": ["2022-01-01", "2023-12-31"],
                                        "frozen_as_of": "2023-12-31", "holdout_evaluated": False,
                                        "risk_thresholds": {"weekly_combination_both_costs": {"sharpe": .25}}},
            "summary.json": {"holdout_evaluated": False, "strategies": [
                {"horizon": 5, "strategy_id": "core4", **prior, "double_cost": prior},
                {"horizon": 5, "strategy_id": "core4_replace_test", **prior, "double_cost": stress,
                 "status": "not_feasible"},
            ]},
            "combination_assessments.json": [
                {"horizon": 5, "strategy_id": "core4_replace_test",
                 "failed_checks": {"train": [], "validation": []}},
                {"horizon": 5, "strategy_id": "core4_replace_rejected",
                 "failed_checks": {"train": ["worse_top_portfolio_tail"], "validation": []}},
            ],
            "definitions_frozen.json": [
                {"candidate_id": "test", "control_id": "x", "expression_text": "ts_mean(x, 10)", "hypothesis": "trial"},
                {"candidate_id": "rejected", "control_id": "y", "expression_text": "ts_mean(y, 10)", "hypothesis": "trial"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in files.items():
                (root / name).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Miner cutoff"):
                load_weekly_feedback(root, "2021-12-31", ["x", "y"])
            result = load_weekly_feedback(root, "2023-12-31", ["x", "y"])
            x = result["by_control"]["x"][0]["trade"]["cost_scenarios"]["double_cost"]
            self.assertEqual(x["failed_thresholds"], ["sharpe"])
            self.assertAlmostEqual(x["delta_vs_core"]["cumulative_return"], -.02)
            y = result["by_control"]["y"][0]
            self.assertIsNone(y["trade"])
            self.assertEqual(y["failed_combination_checks"]["train"], ["worse_top_portfolio_tail"])

    def test_graph_dispatches_only_assigned_control_feedback(self):
        state = {"controls": ["x", "y"], "feedback_by_control": {"x": "private-x", "y": "private-y"},
                 "agent_count": 2, "requested_count": 2, "available": ["x", "y"],
                 "model": "mock", "market_context": "shared"}
        with patch("langgraph_miner.Send", side_effect=lambda _, payload: payload):
            first, second = _fan_out(state)
        self.assertIn("private-x", first["market_context"])
        self.assertNotIn("private-y", first["market_context"])
        self.assertIn("private-y", second["market_context"])
