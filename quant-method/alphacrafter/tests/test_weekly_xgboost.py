import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from weekly_xgboost import fit_weekly_scorer


class WeeklyXGBoostTests(unittest.TestCase):
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
