from dataclasses import asdict
import unittest

from trader_harness import frequency_presets, select_frequency


class TraderHarnessTests(unittest.TestCase):
    def test_compares_feasible_frequencies_and_freezes_config(self):
        window = ("2022-01-01", "2023-12-31")
        trials = [
            {"horizon": h, "config": asdict(config), "status": "feasible_in_proxy",
             "annualized_return": .12, "sharpe": .9, "max_drawdown": -.06,
             "execution_audit": {"liquidation_complete": True,
                                 "turnover_budget_respected": True, "cost_budget_respected": True}}
            for h, config in frequency_presets().items()
        ]
        trials[0]["sharpe"] = 1.1
        trials[0]["execution_audit"]["cost_budget_respected"] = False
        selected = select_frequency(trials, window, evaluated=True)
        self.assertEqual(selected["selected_horizon"], 20)
        self.assertEqual(selected["selected_config"], trials[2]["config"])
        trials[1]["sharpe"] = 1.0
        self.assertEqual(select_frequency(trials, window, evaluated=True)["selected_horizon"], 5)

    def test_combination_can_win_without_single_factor_strategy(self):
        window = ("2022-01-01", "2023-12-31")
        config = asdict(frequency_presets()[5])
        core = {"horizon": 5, "strategy_id": "core4", "members": ["a", "b", "c", "d"],
                "config": config, "status": "feasible_in_proxy", "annualized_return": .12,
                "sharpe": .85, "max_drawdown": -.07,
                "execution_audit": {"liquidation_complete": True,
                                    "turnover_budget_respected": True, "cost_budget_respected": True}}
        core["double_cost"] = {key: core[key] for key in
                               ("annualized_return", "sharpe", "max_drawdown", "execution_audit")}
        replacement = {**core, "strategy_id": "core4_replaced", "sharpe": .95,
                       "members": ["e", "b", "c", "d"]}
        trials = [{"horizon": h, "status": "not_run_no_research_signals"} for h in (1, 20)]
        selected = select_frequency([*trials, core, replacement], window, evaluated=True)
        self.assertEqual(selected["selected_strategy_id"], "core4_replaced")
        self.assertEqual(selected["selected_members"], replacement["members"])

    def test_calibrated_weekly_gate_requires_positive_stress_quality(self):
        from trader_harness import strategy_is_feasible

        audit = {"liquidation_complete": True, "turnover_budget_respected": True,
                 "cost_budget_respected": True}
        metrics = {"annualized_return": .03, "sharpe": .33, "max_drawdown": -.111,
                   "execution_audit": audit}
        trial = {"horizon": 5, "members": ["a", "b", "c", "d"], **metrics}
        self.assertFalse(strategy_is_feasible(trial))
        self.assertTrue(strategy_is_feasible({**trial, "double_cost": metrics}))
        self.assertFalse(strategy_is_feasible({**trial, "double_cost": {**metrics, "annualized_return": -.01}}))
        self.assertFalse(strategy_is_feasible({**trial, "horizon": 20, "double_cost": metrics}))

    def test_cash_when_no_frequency_qualifies(self):
        trials = [{"horizon": h, "status": "not_run_no_research_candidates"} for h in frequency_presets()]
        selected = select_frequency(trials, ("2022-01-01", "2023-12-31"), evaluated=True)
        self.assertIsNone(selected["selected_config"])
        self.assertEqual(selected["status"], "no_feasible_strategy")
