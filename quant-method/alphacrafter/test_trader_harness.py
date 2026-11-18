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

    def test_cash_when_no_frequency_qualifies(self):
        trials = [{"horizon": h, "status": "not_run_no_research_candidates"} for h in frequency_presets()]
        selected = select_frequency(trials, ("2022-01-01", "2023-12-31"), evaluated=True)
        self.assertIsNone(selected["selected_config"])
        self.assertEqual(selected["status"], "no_feasible_strategy")
