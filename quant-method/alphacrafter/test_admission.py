import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from admission import (HORIZON_RESEARCH_POLICIES, ResearchPolicy, assess_weekly_substitution,
                       build_research_pool, research_failures, strategy_plan, validate_trader_pool)
from screener_lite import load_factor_library
from trader_lite import select_configuration


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        metric = {"rows": 100, "mean_daily_rank_ic": .025, "rank_icir": .15,
                  "rank_ic_hit_ratio": .58, "coverage": .95, "turnover": .2}
        self.results = [{"candidate_id": "weak", "accepted": False,
                         "horizons": {h: {s: dict(metric) for s in ("train", "validation", "backtest")} for h in ("1", "5")}}]
        self.definitions = [{"candidate_id": "weak", "op": "feature", "features": ["x"], "hypothesis": "test"}]
        self.audit = {"as_of": "2021-12-31", "label_mode": "execution",
                      "windows": {"train": ["2010-01-01", "2018-12-31"],
                                  "validation": ["2019-01-01", "2021-12-31"]}}

    def pool(self, market="us"):
        return build_research_pool(self.results, self.definitions, self.audit, market)

    def test_weak_signal_is_research_only_and_formal_gate_uses_portfolio(self):
        pool, _ = self.pool()
        self.assertEqual(pool["eligible_counts"], {"1": 1, "5": 1})
        self.assertFalse(pool["candidates_by_horizon"]["1"][0]["strategy_admitted"])
        self.assertEqual(pool["strategy_admitted_count"], 0)
        validate_trader_pool(pool, 1, "2022-01-01")
        good = {"annualized_return": .12, "sharpe": .8, "max_drawdown": -.06}
        self.assertEqual(select_configuration([good]), good)
        self.assertIsNone(select_configuration([{**good, "max_drawdown": -.12}]))

    def test_horizon_direction_is_fit_on_training_not_development(self):
        for split in ("train", "validation"):
            m = self.results[0]["horizons"]["5"][split]
            m.update(mean_daily_rank_ic=-.025, rank_icir=-.15, rank_ic_hit_ratio=.42)
        pool, _ = self.pool()
        self.assertEqual(pool["candidates_by_horizon"]["1"][0]["direction"], 1)
        self.assertEqual(pool["candidates_by_horizon"]["5"][0]["direction"], -1)
        m = self.results[0]["horizons"]["5"]["validation"]
        m.update(mean_daily_rank_ic=.025, rank_icir=.15, rank_ic_hit_ratio=.58)
        pool, _ = self.pool()
        self.assertEqual(pool["eligible_counts"]["5"], 0)

    def test_holdout_metrics_do_not_change_pool_or_assessments(self):
        first = self.pool()
        for h in ("1", "5"):
            self.results[0]["horizons"][h]["backtest"] = {"mean_daily_rank_ic": 1000}
        self.assertEqual(first, self.pool())
        self.assertNotIn("backtest", json.dumps(first))

    def test_missing_nonfinite_and_incomplete_windows_fail_closed(self):
        self.results[0]["horizons"]["1"]["train"]["rank_icir"] = float("nan")
        pool, assessments = self.pool()
        self.assertEqual(pool["eligible_counts"]["1"], 0)
        json.dumps((pool, assessments), allow_nan=False)
        self.audit["as_of"] = "2020-01-01"
        pool, _ = self.pool()
        self.assertEqual(pool["eligible_counts"], {"1": 0, "5": 0})
        self.assertEqual(pool["status"], "pending_complete_development")

    def test_research_thresholds_are_strict_and_finite(self):
        metric = self.results[0]["horizons"]["1"]["train"]
        for key, value, failure in (("mean_daily_rank_ic", .015, "mean_daily_rank_ic"),
                                    ("rank_icir", .1, "rank_icir"), ("rank_ic_hit_ratio", .55, "hit_ratio"),
                                    ("coverage", .9, "coverage"), ("turnover", .4, "turnover")):
            self.assertIn(failure, research_failures({**metric, key: value}, 1, ResearchPolicy()))
        with self.assertRaises(ValueError):
            ResearchPolicy(min_rank_icir=float("nan"))

    def test_frequency_profiles_apply_only_to_their_own_horizon(self):
        for values in self.results[0]["horizons"].values():
            for split in ("train", "validation"):
                values[split]["rank_ic_hit_ratio"] = .54
        pool, _ = build_research_pool(self.results, self.definitions, self.audit,
                                      policies_by_horizon={h: HORIZON_RESEARCH_POLICIES[h] for h in (1, 5)})
        self.assertEqual(pool["eligible_counts"], {"1": 0, "5": 1})
        self.assertEqual(pool["policies_by_horizon"]["5"]["min_hit"], .53)

    def test_replacement_requires_both_costs_and_independent_window(self):
        baseline = {"cumulative_return": .1, "max_drawdown": -.08, "calendar_year_returns": {"2024": .1}}
        improved = {**baseline, "cumulative_return": .11, "calendar_year_returns": {"2024": .11}}
        core = {"1": baseline, "2": baseline}
        self.assertFalse(assess_weekly_substitution(core, {"1": improved, "2": baseline},
                                                   independent=True)["passes_paired_checks"])
        result = assess_weekly_substitution(core, {"1": improved, "2": improved}, independent=False)
        self.assertTrue(result["passes_paired_checks"])
        self.assertFalse(result["replacement_admitted"])

    def test_independent_plan_and_execution_guard(self):
        pool, _ = self.pool("cn")
        plan = strategy_plan(pool, "2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31")
        self.assertEqual(plan["status"], "pending_strategy_definition")
        self.assertFalse(plan["strategy_admitted"])
        with self.assertRaisesRegex(ValueError, "follow research"):
            strategy_plan(pool, "2021-01-01", "2023-12-31", "2024-01-01", "2025-12-31")
        with self.assertRaisesRegex(ValueError, "different execution"):
            validate_trader_pool(pool, 1, "2022-01-01")
        us, _ = self.pool()
        with self.assertRaisesRegex(ValueError, "after research"):
            validate_trader_pool(us, 1, "2021-12-31")

    def test_pool_cannot_be_used_as_screener_library(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pool.json"
            path.write_text(json.dumps(self.pool()[0]))
            with self.assertRaisesRegex(ValueError, "not a trading factor library"):
                load_factor_library(path)

    def test_export_cli_writes_pool_plan_and_no_trading_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename, value in (("metrics.json", self.results), ("candidates.json", self.definitions), ("audit.json", self.audit)):
                (root / filename).write_text(json.dumps(value))
            out = root / "out"
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_research_pool.py")),
                            "--metrics", str(root/"metrics.json"), "--candidates", str(root/"candidates.json"),
                            "--temporal-audit", str(root/"audit.json"), "--market", "cn", "--output-dir", str(out)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((out/"research_pool.json").read_text())["eligible_counts"], {"1": 1, "5": 1})
            self.assertFalse((out/"factor_library.json").exists())
            self.assertFalse((out/"predictions.csv").exists())
            self.assertFalse(json.loads((out/"manifest.json").read_text())["formal_validation_run"])
            medium = [{"candidate_id": "weak", "horizons": {h: self.results[0]["horizons"]["5"] for h in ("5", "20")}}]
            (root/"metrics.json").write_text(json.dumps(medium))
            (root/"audit.json").write_text(json.dumps({**self.audit, "label_mode": "execution_open_open"}))
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_research_pool.py")),
                            "--metrics", str(root/"metrics.json"), "--candidates", str(root/"candidates.json"),
                            "--temporal-audit", str(root/"audit.json"), "--market", "cn", "--horizons", "5,20",
                            "--output-dir", str(root/"medium")], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((root/"medium/research_pool.json").read_text())["eligible_counts"], {"5": 1, "20": 1})

    def test_trader_can_validate_weak_research_factors_on_independent_portfolio_returns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calendar = pd.bdate_range("2022-01-03", "2024-01-10")
            next_date = dict(zip(calendar[:-1], calendar[1:]))
            pd.DataFrame({"date": calendar}).to_csv(root/"calendar.csv", index=False)
            dates = list(pd.bdate_range("2022-01-03", periods=4)) + list(pd.bdate_range("2024-01-02", periods=4))
            rows = []
            for index, date in enumerate(dates):
                for ticker, score in (("A", 1.), ("B", 0.)):
                    rows.append({"date": date, "ticker": ticker, "prediction": score,
                                 "target_open_close": [.01, .02, -.005, .01][index] if index < 4 else -.01,
                                 "label_end_1d": next_date[date]})
            frame = pd.DataFrame(rows)
            frame.drop(columns="prediction").to_csv(root/"features.csv", index=False)
            frame.rename(columns={"date": "datetime", "ticker": "instrument"})[["datetime", "instrument", "prediction"]].to_csv(root/"predictions.csv", index=False)
            (root/"pool.json").write_text(json.dumps(self.pool()[0]))
            command = [sys.executable, str(Path(__file__).with_name("trader_lite.py")),
                       "--input", str(root/"features.csv"), "--predictions", str(root/"predictions.csv"),
                       "--calendar", str(root/"calendar.csv"), "--require-explicit-label-dates",
                       "--research-pool", str(root/"pool.json"), "--configs", ".8,1,1,0",
                       "--selection-start", "2022-01-03", "--selection-end", "2022-01-06",
                       "--backtest-start", "2024-01-02", "--backtest-end", "2024-01-05",
                       "--output-dir", str(root/"out")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((root/"out/summary.json").read_text())
            self.assertEqual(summary["admission_stage"], "formal_strategy_validation")
            self.assertEqual(summary["status"], "selected")
            self.assertLess(summary["backtest"]["cumulative_return"], 0)
            self.assertEqual(summary["factor_gate"], "research_eligibility_not_original_joint_factor_gate")
            self.assertIsNotNone(summary["research_pool"])


if __name__ == "__main__":
    unittest.main()
