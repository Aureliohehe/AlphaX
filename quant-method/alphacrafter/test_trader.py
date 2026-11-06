#!/usr/bin/env python3
"""Regression tests for Trader-lite's explicit long/short portfolio rules."""

from __future__ import annotations

import unittest

import pandas as pd

from trader_lite import TraderConfig, parse_configs, simulate, target_weights


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

    def test_config_parser(self) -> None:
        configs = parse_configs("0.8,0,10,10;0.6,0.5,10,5")
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[1].nshort, 5)


if __name__ == "__main__":
    unittest.main()
