"""Real Qlib dataset/normalization and XGBoost training on synthetic data."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "quant-method"
sys.path[:0] = [str(ROOT / "alphacrafter"), str(ROOT / "machine-learn")]

from train_xgboost_qlib import apply_feature_transform, apply_label_transform, prepare_dataset
from weekly_xgboost import fit_weekly_scorer
from qlib.contrib.model.xgboost import XGBModel


class QlibIntegrationTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("Network forbidden in offline integration"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_real_dataset_normalization_and_xgboost_model(self):
        dates = pd.bdate_range("2021-01-01", periods=30)
        index = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(16)]],
                                           names=["datetime", "instrument"])
        rng = np.random.default_rng(42)
        values = rng.normal(size=(len(index), 2))
        frame = pd.DataFrame(values, index=index,
                             columns=pd.MultiIndex.from_tuples([("feature", "x"), ("feature", "y")]))
        frame[("label", "target")] = values[:, 0] * .1
        normalized = apply_label_transform(apply_feature_transform(frame, "cs_rank"))
        dataset = prepare_dataset(normalized, {
            "train": (dates[0], dates[19]), "valid": (dates[20], dates[24]),
            "test": (dates[25], dates[29]),
        })
        model = XGBModel(max_depth=2, nthread=1, seed=42)
        model.fit(dataset, num_boost_round=5, early_stopping_rounds=2, verbose_eval=False)
        scores = model.predict(dataset, segment="test")
        self.assertEqual(len(scores), 80)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores.std()), 0.)

    def test_weekly_scorer_fits_real_model_with_mature_training_labels(self):
        dates = pd.bdate_range("2021-01-01", periods=35)
        frame = pd.DataFrame([
            {"date": date, "ticker": str(asset), "factor": float(asset),
             "target_open_open_5d": asset * .01,
             "label_end_5d": dates[i + 5] if i + 5 < len(dates) else pd.NaT}
            for i, date in enumerate(dates) for asset in range(32)
        ])
        prediction, _, metadata = fit_weekly_scorer(
            frame, ["factor"], dates[25], dates[-1], training_window=(dates[0], dates[24]))
        self.assertEqual(metadata["training_rows"], 20 * 32)
        self.assertEqual(len(prediction), 10 * 32)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(float(prediction.std()), 0.)


if __name__ == "__main__":
    unittest.main()
