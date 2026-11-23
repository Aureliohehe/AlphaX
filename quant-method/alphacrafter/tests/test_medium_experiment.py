"""Offline checks for medium-frequency experiment records and archive lookup."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_medium_frequency import run
from .test_temporal import run_cli


class MediumExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "market"
        self.source.mkdir()
        self.miner = self.root / "miner"
        self.miner.mkdir()
        (self.miner / "candidates.json").write_text(json.dumps([{
            "candidate_id": "ranked", "op": "feature", "features": ["ranked"],
        }]))
        (self.miner / "registry.json").write_text(json.dumps({
            "registry_version": "miner-lite-v3", "generator": "external_json",
            "as_of": "2021-12-30", "candidate_count": 1,
        }))
        dates = pd.bdate_range("2018-01-02", periods=230).append([
            pd.bdate_range("2021-01-04", periods=230),
            pd.bdate_range("2022-01-03", periods=110),
            pd.bdate_range("2024-01-02", periods=2),
        ])
        (self.source / "daily_pv.h5").write_bytes(b"fixture")
        (self.source / "qlib_day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n")
        tickers = [f"SH60000{i}" for i in range(6)]
        (self.source / "qlib_csi300.txt").write_text("".join(
            f"{ticker}\t2018-01-02\t2025-12-31\n" for ticker in tickers
        ))
        self.membership = pd.DataFrame({"ticker": tickers, "start": pd.Timestamp("2018-01-02"),
                                        "end": pd.Timestamp("2025-12-31")})
        rng = np.random.default_rng(23)
        self.bars = []
        for index, ticker in enumerate(tickers):
            prices = 100 * np.exp(np.cumsum(.001 * index + rng.normal(0, .001, size=len(dates))))
            self.bars.append(pd.DataFrame({
                "date": dates, "ticker": ticker, "open": prices, "close": prices,
                "high": prices * 1.01, "low": prices * .99, "volume": 1000., "ranked": index,
            }))
        self.bars = pd.concat(self.bars, ignore_index=True)
        self.feature_panel = self.root / "panel.csv"
        self.bars[["date", "ticker", "ranked"]].to_csv(self.feature_panel, index=False)

    def options(self, name, **options):
        defaults = {
            "output_dir": self.root / name, "miner_dir": self.miner, "feature_panel": self.feature_panel,
            "evaluate_strategies": False, "factor_strength_rule": "all", "compare_weekly_xgboost": False,
            "compare_weekly_xgboost_additions": False, "weekly_xgboost_augmentation_weight": 1.,
            "weekly_xgboost_augmentation_method": "refit",
        }
        return Namespace(**(defaults | options))

    def fixture_bars(self, source, membership, start, end):
        return self.bars[self.bars.date.between(start, end)], self.membership

    def fixture_quotes(self, source, tickers, calendar, start, end):
        columns = ["date", "ticker", "open", "close"]
        first = calendar[calendar < pd.Timestamp(start)][-1]
        quotes = self.bars.loc[self.bars.date.between(first, end) & self.bars.ticker.isin(tickers), columns]
        benchmark = pd.DataFrame({"date": quotes.date.unique(), "ticker": "SH000300", "open": 100., "close": 100.})
        return pd.concat([quotes, benchmark], ignore_index=True)

    def test_modes_preserve_research_metrics_and_catalog_archived_results(self):
        research = self.options("research")
        strategy = self.options("strategy", evaluate_strategies=True)
        with patch("run_medium_frequency.load_member_bars", side_effect=self.fixture_bars) as load, \
             patch("run_medium_frequency.load_quotes", side_effect=self.fixture_quotes):
            factor_summary = run(research, self.source)
            strategy_summary = run(strategy, self.source)
        self.assertEqual([call.args[-1] for call in load.call_args_list], ["2021-12-31", "2023-12-31"])
        self.assertEqual(factor_summary["research_eligible"], strategy_summary["research_eligible"])
        self.assertEqual((research.output_dir / "factor_metrics.json").read_bytes(),
                         (strategy.output_dir / "factor_metrics.json").read_bytes())
        self.assertTrue(any("cumulative_return" in trial for trial in strategy_summary["strategies"]))

        archive = self.root / "archived"
        strategy.output_dir.rename(archive)
        records = json.loads(run_cli("list_experiments.py", self.root).stdout)
        by_mode = {record["run_mode"]: record for record in records}
        factor = by_mode["factor_research"]
        comparison = by_mode["strategy_comparison"]
        self.assertEqual(factor["strategy_status"], "not_evaluated")
        self.assertEqual(comparison["experiment_type"], "medium_frequency_research")
        self.assertEqual(comparison["status"], "completed")
        self.assertEqual(comparison["miner_source"], {"generator": "external_json", "as_of": "2021-12-30"})
        selection = json.loads((archive / "strategy_selection.json").read_text())
        self.assertEqual(comparison["strategy_status"], selection["status"])
        self.assertEqual(comparison["metrics"]["selected_horizon"], selection["selected_horizon"])
        self.assertEqual(comparison["directory"], str(archive))
        self.assertNotEqual(comparison["experiment_fingerprint"], factor["experiment_fingerprint"])
        self.assertTrue(comparison["strategy_iteration_previously_examined"])
        self.assertEqual(comparison["final_reserved"], {"start": "2024-01-01", "end": "2025-12-31"})
        self.assertIsNone(comparison["holdout"])
        self.assertFalse(comparison["metrics"]["holdout_evaluated"])
        self.assertEqual(set(comparison["metrics"]["by_horizon"]), {"1", "5", "20"})
        for horizon, count in strategy_summary["research_eligible"].items():
            self.assertEqual(comparison["metrics"]["by_horizon"][horizon]["eligible_factors"], count)
        self.assertTrue({"report.md", "strategy_selection.json"}.issubset(comparison["artifacts"]))
        self.assertTrue(all((archive / path).is_file() for path in comparison["artifacts"]))

    def test_failed_stage_keeps_input_and_xgboost_settings(self):
        args = self.options("failed", evaluate_strategies=True, compare_weekly_xgboost=True)
        with patch("run_medium_frequency.load_member_bars", side_effect=ValueError("invalid test bars")):
            with self.assertRaisesRegex(ValueError, "invalid test bars"):
                run(args, self.source)
        record, = json.loads(run_cli("list_experiments.py", args.output_dir).stdout)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_stage"], "building_features")
        self.assertEqual(record["weekly_xgboost"], {"comparison": True, "additions": False})
        self.assertTrue(record["run_id"])
        self.assertIn("definitions_frozen.json", record["artifacts"])
        self.assertNotIn("report.md", record["artifacts"])
        self.assertIsNone(record["metrics"]["holdout_evaluated"])

    def test_report_write_failure_does_not_publish_completed_status(self):
        args = self.options("failed_report")
        write_text = Path.write_text

        def write(path, *values, **kwargs):
            if path == args.output_dir / "report.md":
                status = json.loads((args.output_dir / "status.json").read_text())
                self.assertNotEqual(status["status"], "completed_research")
                raise OSError("fixture report write failure")
            return write_text(path, *values, **kwargs)

        with patch("run_medium_frequency.load_member_bars", side_effect=self.fixture_bars), \
             patch.object(Path, "write_text", write):
            with self.assertRaisesRegex(OSError, "fixture report write failure"):
                run(args, self.source)
        record, = json.loads(run_cli("list_experiments.py", args.output_dir).stdout)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failed_stage"], "finalizing")
        self.assertIn("summary.json", record["artifacts"])
        self.assertIsNone(record["metrics"]["holdout_evaluated"])
