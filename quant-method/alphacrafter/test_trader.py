#!/usr/bin/env python3
"""Regression tests for Trader-lite's explicit long/short portfolio rules."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trader_lite import (
    TraderConfig, is_feasible, load_inputs, parse_configs, select_configuration,
    simulate, summarize, target_weights, validate_intervals,
)


class TraderTests(unittest.TestCase):
    def test_long_short_exposures_follow_beta_and_gamma(self) -> None:
        group = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "prediction": [0.4, 0.2, -0.1, -0.3],
            }
        )
        weights = target_weights(group, TraderConfig(0.8, 0.25, 2, 2))
        self.assertAlmostEqual(sum(value for value in weights.values() if value > 0), 0.5)
        self.assertAlmostEqual(-sum(value for value in weights.values() if value < 0), 0.3)

    def test_validation_simulation_uses_explicit_positions(self) -> None:
        dates = pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["A", "B", "C", "D"] * 2,
                "prediction": [0.4, 0.2, -0.1, -0.3] * 2,
                "target_open_close": [0.01, 0.01, -0.01, -0.01] * 2,
            }
        )
        report, positions = simulate(
            frame, TraderConfig(0.8, 0.0, 1, 1), "2020-01-01", "2020-01-02", 0.0
        )
        self.assertEqual(len(report), 2)
        self.assertFalse(positions.empty)
        self.assertGreater(report["net_return"].iloc[0], 0.0)

    def panel(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-12-28"] * 4 + ["2023-12-29"] * 4),
            "ticker": ["A", "B", "C", "D"] * 2,
            "prediction": [4., 3., 2., 1.] * 2,
            "target_open_close": [0.] * 8,
        })

    def test_intraday_cost_is_paid_both_legs_every_day(self) -> None:
        report, _ = simulate(self.panel(), TraderConfig(.8, 0, 1, 1), "2023-01-01", "2023-12-31", .0001)
        np.testing.assert_allclose(report["traded_notional"], 1.6)
        np.testing.assert_allclose(report["transaction_cost"], .00016)
        np.testing.assert_allclose(report["net_return"], -.00016)

    def test_exit_cost_tracks_changed_price(self) -> None:
        frame = self.panel()
        frame.loc[frame["ticker"] == "A", "target_open_close"] = .1
        report, _ = simulate(frame, TraderConfig(.8, 1, 1, 0), "2023-01-01", "2023-12-31", .001)
        self.assertAlmostEqual(report["traded_notional"].iloc[0], .8 + .88)

    def test_missing_future_label_does_not_change_stock_selection(self) -> None:
        frame = self.panel()
        expected = target_weights(frame.iloc[:4], TraderConfig(.8, 0, 1, 1))
        frame.loc[frame["ticker"] == "A", "target_open_close"] = np.nan
        self.assertEqual(target_weights(frame.iloc[:4], TraderConfig(.8, 0, 1, 1)), expected)
        with self.assertRaisesRegex(ValueError, "execution return"):
            simulate(frame, TraderConfig(.8, 0, 1, 1), "2023-01-01", "2023-12-31", 0.)

    def test_missing_signal_goes_to_cash_not_stale_weights(self) -> None:
        frame = self.panel()
        frame.loc[4:, "prediction"] = np.nan
        report, positions = simulate(frame, TraderConfig(.8, 0, 1, 1), "2023-01-01", "2023-12-31", .0001)
        self.assertEqual(report["selected_assets"].tolist(), [2, 0])
        self.assertEqual(report["net_return"].iloc[-1], 0)
        self.assertEqual(len(positions), 2)

    def test_short_only_and_long_only_do_not_reserve_inactive_names(self) -> None:
        group = self.panel().iloc[:4]
        self.assertEqual(target_weights(group, TraderConfig(.8, -1, 0, 1)), {"D": -.8})
        self.assertEqual(target_weights(group, TraderConfig(.8, 1, 1, 0)), {"A": .8})

    def test_ties_are_independent_of_input_order(self) -> None:
        group = self.panel().iloc[:4].copy()
        group["prediction"] = 1.
        config = TraderConfig(.8, 0, 1, 1)
        self.assertEqual(target_weights(group, config), target_weights(group.iloc[::-1], config))

    def test_initial_loss_counts_in_drawdown_and_sharpe_uses_risk_free(self) -> None:
        report, _ = simulate(self.panel(), None, "2023-01-01", "2023-12-31", 0.)
        report["net_return"] = [-.1, .02]
        metrics = summarize(report, 100.)
        self.assertAlmostEqual(metrics["max_drawdown"], -.1)
        daily_rf = 1.0381 ** (1 / 252) - 1
        expected = ((-.1 + .02) / 2 - daily_rf) / pd.Series([-.1, .02]).std() * np.sqrt(252)
        self.assertAlmostEqual(metrics["sharpe"], expected)

    def test_no_feasible_strategy_stays_in_cash_and_serializes_strict_json(self) -> None:
        results = [{"annualized_return": .5, "sharpe": 2., "max_drawdown": -.1}]
        self.assertIsNone(select_configuration(results))
        report, positions = simulate(self.panel(), None, "2023-01-01", "2023-12-31", .0001)
        metrics = summarize(report, 100.)
        self.assertEqual(metrics["cumulative_return"], 0.)
        self.assertIsNone(metrics["sharpe"])
        self.assertTrue(positions.empty)
        json.dumps(metrics, allow_nan=False)

    def test_gate_is_strict_finite_and_maximizes_only_feasible_sharpe(self) -> None:
        good = {"annualized_return": .1, "sharpe": .7, "max_drawdown": -.05}
        for key, boundary in (("annualized_return", .08), ("sharpe", .6), ("max_drawdown", -.08)):
            self.assertFalse(is_feasible({**good, key: boundary}))
            self.assertFalse(is_feasible({**good, key: np.nan}))
        self.assertFalse(is_feasible({**good, "sharpe": None}))
        bad = {**good, "sharpe": 3., "max_drawdown": -.2}
        self.assertEqual(select_configuration([bad, good]), good)

    def test_trial_budget_configs_and_split_validation(self) -> None:
        with self.assertRaises(ValueError):
            parse_configs(".8,0,1,1; .7,0,1,1; .6,0,1,1; .5,0,1,1")
        for values in ((float("nan"), 0, 1, 1), (.8, 2, 1, 1), (.8, 0, 0, 1), (.8, 0, 1.5, 1)):
            with self.assertRaises(ValueError):
                TraderConfig(*values)
        with self.assertRaises(ValueError):
            validate_intervals("2023-01-01", "2024-01-02", "2024-01-02", "2025-01-01")

    def test_execution_date_purges_unsettled_labels_at_split_boundary(self) -> None:
        frame = self.panel()
        frame["execution_date"] = pd.to_datetime(["2023-12-29"] * 4 + ["2024-01-02"] * 4)
        report, _ = simulate(frame, TraderConfig(.8, 0, 1, 1), "2023-01-01", "2023-12-29", 0.)
        changed = frame.copy()
        changed.loc[4:, "target_open_close"] = 100.
        other, _ = simulate(changed, TraderConfig(.8, 0, 1, 1), "2023-01-01", "2023-12-29", 0.)
        pd.testing.assert_frame_equal(report, other)
        self.assertEqual(len(report), 1)

    def test_cli_no_feasible_strategy_and_duplicate_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.concat([self.panel(), self.panel().assign(date=lambda x: x["date"] + pd.DateOffset(years=1))])
            features = root / "features.csv"
            predictions = root / "predictions.csv"
            frame.drop(columns="prediction").to_csv(features, index=False)
            frame.rename(columns={"date": "datetime", "ticker": "instrument"})[["datetime", "instrument", "prediction"]].to_csv(predictions, index=False)
            loaded = load_inputs(predictions, features)
            self.assertEqual(loaded["execution_date"].iloc[0], pd.Timestamp("2023-12-29"))
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("trader_lite.py")),
                "--input", str(features), "--predictions", str(predictions),
                "--output-dir", str(root / "out"), "--configs", ".8,0,1,1",
                "--backtest-end", "2024-12-31",
            ], check=True, capture_output=True, text=True)
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["status"], "no_feasible_strategy")
            self.assertIsNone(summary["selected_config"])
            self.assertTrue(pd.read_csv(root / "out/positions.csv").empty)
            duplicated = pd.read_csv(predictions)
            pd.concat([duplicated, duplicated.iloc[:1]]).to_csv(predictions, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_inputs(predictions, features)

    def test_config_parser(self) -> None:
        configs = parse_configs("0.8,0,10,10;0.6,0.5,10,5")
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[1].nshort, 5)


if __name__ == "__main__":
    unittest.main()
