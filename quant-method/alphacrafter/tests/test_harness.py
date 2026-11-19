#!/usr/bin/env python3
"""Small regression tests for Miner-lite temporal and cost boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from miner_lite import (
    Candidate,
    META_COLUMNS,
    candidate_from_dict,
    derive_target_5d,
    build_accepted_signal,
    evaluate_candidate_score,
    evaluate_score,
    expression_novelty_report,
    library_factor,
    load_candidates,
    needs_revalidation,
    parse_llm_candidates,
)
from screener_lite import cross_section_rank, ensemble_signal
from run_agent_execution_demo import demo_gate
from run_miner_walk_forward import (
    build_signal,
    select_factors,
    validation_portfolio_metrics,
)


class MinerHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.Series(
            pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2021-01-04",
                    "2021-01-04",
                    "2021-01-04",
                ]
            )
        )
        self.frame = pd.DataFrame(
            {
                "date": self.dates,
                "ticker": ["A", "B", "C", "A", "B", "C"],
            }
        )
        self.ranked = pd.DataFrame({"feature_x": [0.1, 0.5, 0.9, 0.2, 0.4, 0.8]})
        self.frame["feature_x"] = self.ranked["feature_x"]
        self.target = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        self.candidates = [
            Candidate("single__feature_x", "feature", ("feature_x",), "test")
        ]

    def test_future_target_changes_do_not_change_train_selection(self) -> None:
        selected = select_factors(
            self.dates,
            self.target,
            self.frame,
            self.candidates,
            "2020-12-31",
            0.0,
        )
        changed_target = self.target.copy()
        changed_target.iloc[3:] = [100.0, -100.0, 50.0]
        changed = select_factors(
            self.dates,
            changed_target,
            self.frame,
            self.candidates,
            "2020-12-31",
            0.0,
        )
        self.assertEqual(selected[0]["direction"], changed[0]["direction"])
        self.assertEqual(selected[0]["train_rank_ic"], changed[0]["train_rank_ic"])

    def test_signal_is_truncated_at_evaluation_end(self) -> None:
        selected = [
            {
                "candidate_id": "single__feature_x",
                "op": "feature",
                "features": ["feature_x"],
                "direction": 1,
            }
        ]
        signal = build_signal(self.frame, self.frame, selected, "2020-12-31")
        self.assertEqual(signal["datetime"].max(), pd.Timestamp("2020-01-02"))

    def test_validation_cost_reduces_net_return(self) -> None:
        low_cost = validation_portfolio_metrics(
            self.dates,
            self.frame["ticker"],
            self.target,
            self.ranked["feature_x"],
            "2020-01-01",
            "2021-12-31",
            2,
            1,
            0.0,
        )
        high_cost = validation_portfolio_metrics(
            self.dates,
            self.frame["ticker"],
            self.target,
            self.ranked["feature_x"],
            "2020-01-01",
            "2021-12-31",
            2,
            1,
            0.01,
        )
        self.assertLess(high_cost["validation_cumulative_return"], low_cost["validation_cumulative_return"])

    def test_future_label_is_metadata(self) -> None:
        self.assertIn("target_open_close", META_COLUMNS)
        self.assertIn("target_5d", META_COLUMNS)

    def test_five_day_target_uses_only_future_sessions(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["A"] * 7,
                "date": pd.date_range("2020-01-01", periods=7),
                "return_1d": [0.01] * 7,
            }
        )
        target = derive_target_5d(frame)
        self.assertAlmostEqual(float(target.iloc[0]), (1.01 ** 5) - 1.0)
        self.assertTrue(pd.isna(target.iloc[-1]))

    def test_safe_time_series_expression_is_evaluated(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "ticker": ["A"] * 5,
                "feature_x": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        ranked = frame[["feature_x"]].groupby(frame["date"]).rank(pct=True)
        candidate = candidate_from_dict(
            {
                "candidate_id": "ts_delta_feature_x",
                "features": ["feature_x"],
                "expression": {"op": "ts_delta", "feature": "feature_x", "window": 2},
            }
        )
        score = evaluate_candidate_score(candidate, frame)
        self.assertTrue(pd.isna(score.iloc[0]))
        self.assertEqual(float(score.iloc[2]), 2.0)

    def test_factor_library_revalidation_interval(self) -> None:
        factor = {
            "candidate_id": "single__feature_x",
            "op": "feature",
            "features": ["feature_x"],
            "direction": 1,
        }
        stored = library_factor(factor, None, pd.Timestamp("2020-01-01"), 90)
        self.assertFalse(needs_revalidation(stored, pd.Timestamp("2020-03-30"), 90))
        self.assertTrue(needs_revalidation(stored, pd.Timestamp("2020-04-01"), 90))
        self.assertEqual(stored["validation_count"], 1)
        self.assertEqual(len(stored["validation_history"]), 1)

    def test_execution_demo_gate_does_not_silently_use_paper_gate_or_failures(self) -> None:
        row = {"direction_from_train": 1, "horizons": {"1": {
            split: {"mean_daily_ic": .001, "mean_daily_rank_ic": .001,
                    "icir": .01, "rank_icir": .01, "rank_ic_hit_ratio": .51,
                    "coverage": 1., "turnover": .2} for split in ("train", "validation")}}}
        self.assertTrue(demo_gate(row))
        self.assertFalse(demo_gate({**row, "direction_from_train": 0}))
        changed = json.loads(json.dumps(row))
        changed["horizons"]["1"]["validation"]["rank_ic_hit_ratio"] = .49
        self.assertFalse(demo_gate(changed))
        changed["horizons"]["1"]["validation"]["rank_ic_hit_ratio"] = .51
        changed["horizons"]["1"]["train"]["icir"] = -0.01
        self.assertFalse(demo_gate(changed))

    def test_llm_repairs_only_redundant_ast_features_and_rejects_invalid_ops(self) -> None:
        audit = []
        candidates = parse_llm_candidates([
            {"candidate_id": "llm_safe", "features": ["wrong"], "hypothesis": "fixture",
             "expression": {"op": "feature", "feature": "feature_a"}},
            {"candidate_id": "llm_unsafe", "features": ["feature_a"],
             "expression": {"op": "exec", "input": "open('/tmp/unsafe')"}},
        ], ["feature_a"], audit)
        self.assertEqual([item.candidate_id for item in candidates], ["llm_safe"])
        self.assertEqual(candidates[0].features, ("feature_a",))
        self.assertEqual([entry["status"] for entry in audit], ["normalized_features", "rejected"])
        with self.assertRaisesRegex(ValueError, "no valid"):
            parse_llm_candidates([{"candidate_id": "broken", "op": "eval", "features": ["feature_a"]}], ["feature_a"], [])

    def test_curated_alpha158_candidates_are_explicit_and_not_proxy_regressions(self) -> None:
        source = Path(__file__).resolve().parents[1].joinpath("candidates_alpha158_curated_v1.json")
        payload = json.loads(source.read_text())
        self.assertEqual(len(payload), 18)
        self.assertEqual(len({item["candidate_id"] for item in payload}), 18)
        candidates = load_candidates(source, [item["features"][0] for item in payload])
        self.assertTrue(all(candidate.expression_spec == {"op": "feature", "feature": candidate.features[0]}
                            for candidate in candidates))
        self.assertFalse(any(any(proxy in candidate.features[0] for proxy in ("beta", "rsqr", "resi", "imax", "imin", "imxd", "vwap", "rank"))
                             for candidate in candidates))

    def test_expression_novelty_report_is_structural(self) -> None:
        report = expression_novelty_report(
            [
                Candidate("a", "feature", ("feature_a",), ""),
                Candidate("b", "difference", ("feature_a", "feature_b"), ""),
            ]
        )
        self.assertEqual(report["candidate_count"], 2)
        self.assertGreater(report["internal_structural_diversity"], 0.0)

    def test_coverage_and_turnover_are_reported(self) -> None:
        dates = pd.Series(pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3))
        instruments = pd.Series(["A", "B", "C"] * 2)
        score = pd.Series([0.1, 0.5, float("nan"), 0.2, 0.4, 0.8])
        target = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        metric = evaluate_score(
            dates, score, target, pd.Series([True] * len(dates)), instruments
        )
        self.assertAlmostEqual(metric.coverage, 5.0 / 6.0)
        self.assertGreaterEqual(metric.turnover, 0.0)
        self.assertLessEqual(metric.turnover, 1.0)
        self.assertTrue(hasattr(metric, "mean_daily_ic"))
        self.assertTrue(hasattr(metric, "icir"))

    def test_icir_uses_paper_population_standard_deviation(self) -> None:
        dates = pd.Series(pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4 + ["2020-01-06"] * 4))
        score = pd.Series([1., 2., 3., 4.] * 3)
        target = pd.Series([1., 2., 3., 4., 1., 2., 4., 3., 3., 2., 4., 1.])
        metric = evaluate_score(dates, score, target, pd.Series(True, index=dates.index))
        daily_ic = pd.Series([score.iloc[i:i + 4].corr(target.iloc[i:i + 4]) for i in (0, 4, 8)])
        self.assertAlmostEqual(metric.daily_ic_std, daily_ic.std(ddof=0))
        self.assertAlmostEqual(metric.icir, daily_ic.mean() / daily_ic.std(ddof=0))
        self.assertNotAlmostEqual(metric.icir, daily_ic.mean() / daily_ic.std(ddof=1))

    def test_raw_vs_ranked_single_feature_changes_pearson_not_rank_ic(self) -> None:
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
            "ticker": ["A", "B", "C", "D"] * 2,
            "signal": [1., 2., 3., 30., 1., 4., 6., 40.],
        })
        target = pd.Series([.01, .02, .04, .06, .01, .05, .03, .07])
        ranked = frame[["signal"]].groupby(frame["date"]).rank(pct=True)
        legacy = Candidate("ranked", "expression", ("signal",), "",
                           {"op": "rank", "input": {"op": "feature", "feature": "signal"}})
        raw = Candidate("raw", "expression", ("signal",), "", {"op": "feature", "feature": "signal"})
        rank_metric = evaluate_score(frame["date"], evaluate_candidate_score(legacy, frame),
                                     target, pd.Series(True, index=frame.index), frame["ticker"])
        raw_metric = evaluate_score(frame["date"], evaluate_candidate_score(raw, frame),
                                    target, pd.Series(True, index=frame.index), frame["ticker"])
        self.assertNotAlmostEqual(raw_metric.mean_daily_ic, rank_metric.mean_daily_ic)
        self.assertAlmostEqual(raw_metric.mean_daily_rank_ic, rank_metric.mean_daily_rank_ic)

    def test_legacy_and_ast_share_scores_metrics_and_rendering(self) -> None:
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03"] * 4),
            "ticker": ["A", "B", "C", "D"] * 2,
            "x": [1., 2., 30., 4., 1., 4., 60., 8.],
            "y": [10., 3., 2., 1., 9., 4., 1., 3.],
        })
        target = pd.Series([.01, .02, .04, .06, .01, .05, .03, .07])
        for op in ("feature", "sum", "difference"):
            with self.subTest(op=op):
                fields = ["x"] if op == "feature" else ["x", "y"]
                legacy = candidate_from_dict({"candidate_id": "legacy", "op": op, "features": fields})
                spec = ({"op": "feature", "feature": "x"} if op == "feature" else
                        {"op": op, "inputs": [{"op": "feature", "feature": f} for f in fields]})
                ast = candidate_from_dict({"candidate_id": "ast", "expression": spec})
                left, right = (evaluate_candidate_score(c, frame) for c in (legacy, ast))
                expected = frame.x if op == "feature" else frame.x + frame.y if op == "sum" else frame.x - frame.y
                pd.testing.assert_series_equal(left, expected)
                pd.testing.assert_series_equal(left, right)
                self.assertEqual(legacy.expression, ast.expression)
                metrics = [evaluate_score(frame.date, score, target, pd.Series(True, index=frame.index), frame.ticker)
                           for score in (left, right)]
                self.assertEqual(metrics[0], metrics[1])

    def test_composite_ranks_after_expression_and_preserves_missing_components(self) -> None:
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 3),
                              "ticker": ["A", "B", "C"], "x": [1., 2., 100.], "y": [3., 1., float("nan")]})
        factors = [{"candidate_id": f, "op": "feature", "features": [f], "direction": 1} for f in ("x", "y")]
        signal = build_accepted_signal(frame, factors)
        expected = (frame[["x", "y"]].rank(pct=True) - .5).sum(axis=1, min_count=2) / 2
        pd.testing.assert_series_equal(signal.prediction, expected, check_names=False)
        screened = ensemble_signal(
            {f: cross_section_rank(frame[f], frame.date) for f in ("x", "y")},
            [{"candidate_id": f, "direction": 1, "weight": .5} for f in ("x", "y")], frame.index,
        )
        pd.testing.assert_series_equal(signal.prediction, screened, check_names=False)

    def test_accepted_ast_factor_is_rehydrated_for_signal_building(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3
                ),
                "ticker": ["A", "B", "C"] * 3,
                "feature_x": [1.0, 2.0, 3.0, 2.0, 4.0, 8.0, 3.0, 6.0, 12.0],
            }
        )
        ranked = frame[["feature_x"]].groupby(frame["date"]).rank(pct=True)
        accepted = [
            {
                "candidate_id": "ts_delta_feature_x",
                "op": "expression",
                "features": ["feature_x"],
                "expression_spec": {
                    "op": "ts_delta",
                    "feature": "feature_x",
                    "window": 2,
                },
                "direction": 1,
            }
        ]
        signal = build_accepted_signal(frame, accepted)
        self.assertEqual(len(signal), len(frame))
        self.assertTrue(signal["prediction"].notna().any())


if __name__ == "__main__":
    unittest.main()
