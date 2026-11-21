import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from cn_portfolio import CNConfig
from trader_harness import AUDIT_CHECKS
from weekly_xgboost import evaluate_factor_additions, fit_weekly_scorer, residual_factor_rank


class WeeklyXGBoostTests(unittest.TestCase):
    def test_residual_rank_projects_out_core_using_only_the_signal_date(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-04"] * 4 + ["2022-01-05"] * 4),
            "core": [1., 2., 3., 4.] * 2,
            "variant": [10., 20., 30., 40., 1., 4., 2., 3.],
        })
        scores = residual_factor_rank(frame, ["core"], "variant")
        self.assertEqual(scores.iloc[:4].tolist(), [0.] * 4)
        self.assertGreater(scores.iloc[4:].abs().sum(), 0.)
        self.assertAlmostEqual((scores.iloc[4:] * frame.core.iloc[4:]).sum(), 0.)
        changed = frame.copy()
        changed.loc[4:, "variant"] = 10.
        pd.testing.assert_series_equal(scores.iloc[:4], residual_factor_rank(changed, ["core"], "variant").iloc[:4])

    def test_addition_and_core_use_the_same_complete_rows(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(["2021-12-31", "2022-01-04", "2022-01-05"]))
        frame = pd.DataFrame({"date": [pd.Timestamp("2021-12-22")] * 2 + [calendar[0]] * 3,
                              "ticker": ["SH1", "SH2", "SH1", "SH2", "SH3"],
                              "core": [.1, .2, .1, .2, .3], "extra": [.5, .4, .2, .3, None],
                              "duplicate": [1., 2., 3., 2., 1.],
                              "label_end_5d": [calendar[0]] * 2 + [calendar[2]] * 3,
                              "target_open_open_5d": [.01] * 5})
        config = CNConfig(1, rebalance_interval=5)
        seen = []

        def fit(common, columns, first, end, *, training_window):
            seen.append((common.index.tolist(), columns))
            scores = [.2, .1] if len(columns) == 1 else [.1, .2]
            return pd.Series(scores, index=[2, 3]), None, {"training_rows": 2}

        report = pd.DataFrame({"net_return": [.01, .02], "gross_return": [.01, .02],
                               "benchmark_return": [.01, .01], "turnover": [.1, .1]}, index=calendar[1:])
        execution = dict.fromkeys(AUDIT_CHECKS, True)
        candidate = {"candidate_id": "extra", "expression_text": "ts_mean(core, 10)"}
        duplicate = {"candidate_id": "duplicate", "expression_text": "rank(core)"}
        with patch("weekly_xgboost.fit_weekly_scorer", side_effect=fit), patch(
            "cn_portfolio.simulate_cn", return_value=(report, None, None, execution)
        ) as simulator:
            row, skipped = evaluate_factor_additions(
                frame, {"core": "core", "extra": "extra", "duplicate": "duplicate"}, ["core"],
                [candidate, duplicate], calendar, None, config, ("2022-01-04", "2022-01-05"),
                augmentation_weight=.25,
            )
        self.assertEqual(seen, [([0, 1, 2, 3], ["core"]), ([0, 1, 2, 3], ["core", "extra"])])
        self.assertTrue(all(call.args[0].ticker.tolist() == ["SH1", "SH2"] for call in simulator.call_args_list))
        self.assertEqual(simulator.call_args_list[2].args[0].prediction.tolist(), [.875, .625])
        self.assertEqual(row["augmentation_weight"], .25)
        self.assertIsNone(row["control"])
        self.assertFalse(row["qualifies_in_replay"])
        self.assertEqual(skipped["equivalent_to"], "core")
        self.assertEqual(skipped["status"], "rank_duplicate_in_training")
        seen.clear()
        with patch("weekly_xgboost.fit_weekly_scorer", side_effect=fit), patch(
            "cn_portfolio.simulate_cn", return_value=(report, None, None, execution)
        ) as simulator:
            direct, = evaluate_factor_additions(
                frame, {"core": "core", "extra": "extra"}, ["core"], [candidate],
                calendar, None, config, ("2022-01-04", "2022-01-05"),
                augmentation_weight=.25, augmentation_method="factor_rank",
            )
            inactive, = evaluate_factor_additions(
                frame, {"core": "core", "extra": "extra"}, ["core"], [candidate],
                calendar, None, config, ("2022-01-04", "2022-01-05"),
                augmentation_weight=.25, augmentation_method="factor_rank",
                augmentation_active=pd.Series(False, index=[calendar[0]]),
            )
        self.assertEqual(seen, [([0, 1, 2, 3], ["core"])] * 2)
        self.assertEqual(simulator.call_args_list[2].args[0].prediction.tolist(), [.875, .625])
        self.assertEqual(direct["augmentation_method"], "factor_rank")
        self.assertEqual(direct["active_signal_dates"], 1)
        self.assertEqual(simulator.call_args_list[6].args[0].prediction.tolist(), [.2, .1])
        self.assertEqual(inactive["active_signal_dates"], 0)

    def test_late_labels_and_comparison_window_never_enter_fit(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2021-12-23", "2021-12-23", "2021-12-27",
                                    "2021-12-31", "2022-01-04"]),
            "label_end_5d": pd.to_datetime(["2021-12-31", "2021-12-31", "2022-01-05",
                                            "2022-01-10", "2022-01-11"]),
            "target_open_open_5d": [.01, .02, .99, .99, .99],
            "feature_a": [1., 2., 3., 4., 5.],
            "feature_b": [2., 1., 3., 4., None],
        })
        model = Mock()
        model.predict.return_value = [.25]
        xgboost = SimpleNamespace(XGBRegressor=Mock(return_value=model), __version__="fixture")
        with patch.dict("sys.modules", {"xgboost": xgboost}):
            scores, _, metadata = fit_weekly_scorer(frame, ["feature_a", "feature_b"],
                                                   "2021-12-31", "2022-01-04")
        inputs, labels = model.fit.call_args.args
        self.assertEqual(inputs.index.tolist(), [0, 1])
        self.assertEqual(labels.index.tolist(), [0, 1])
        self.assertEqual(metadata["last_label_end"], "2021-12-31")
        self.assertEqual(scores.to_dict(), {3: .25})

    def test_refuses_a_training_cutoff_after_the_first_signal(self):
        with patch.dict("sys.modules", {"xgboost": SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError, "first signal"):
                fit_weekly_scorer(pd.DataFrame(), [], "2021-12-31", "2022-01-04",
                                  training_window=("2020-01-01", "2022-01-01"))
