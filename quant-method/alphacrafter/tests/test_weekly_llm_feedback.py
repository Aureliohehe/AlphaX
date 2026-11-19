import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langgraph_miner import _fan_out
from weekly_llm_feedback import load_weekly_feedback


class WeeklyFeedbackTests(unittest.TestCase):
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
