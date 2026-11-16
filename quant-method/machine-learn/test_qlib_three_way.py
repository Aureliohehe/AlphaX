"""Three-way backtest: split-purge and signal preparation without Qlib installed."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_qlib_three_way import BASE_NAMES, CUSTOM, SETS, prepare, purge


class ThreeWayTests(unittest.TestCase):
    def panel(self):
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2020-12-30", "2020-12-31", "2021-01-04", "2021-12-30",
            "2021-12-31", "2022-01-03", "2022-01-04",
        ]))
        index = pd.MultiIndex.from_product([["A", "B"], calendar], names=["instrument", "datetime"])
        frame = pd.DataFrame({"signal": range(len(index)), "LABEL": [.01] * len(index)}, index=index)
        return frame, calendar

    def test_split_end_labels_cannot_cross_boundary(self):
        frame, calendar = self.panel()
        train = purge(frame, calendar, ("2018-01-02", "2020-12-31"))
        valid = purge(frame, calendar, ("2021-01-04", "2021-12-31"))
        self.assertEqual(set(train.index.get_level_values("datetime")), {pd.Timestamp("2020-12-30")})
        self.assertEqual(set(valid.index.get_level_values("datetime")), {pd.Timestamp("2021-01-04"), pd.Timestamp("2021-12-30")})

    def test_inference_keeps_rows_without_future_label(self):
        frame, calendar = self.panel()
        frame.loc[("A", pd.Timestamp("2022-01-04")), "LABEL"] = np.nan
        train, valid, inference = prepare(frame, ["signal"], calendar)
        self.assertEqual(len(train[0]), 2)
        self.assertEqual(len(valid[0]), 4)
        self.assertIn(("A", pd.Timestamp("2022-01-04")), inference.index)
        self.assertEqual(len(BASE_NAMES), 20)
        self.assertEqual(len(CUSTOM), 2)
        self.assertEqual(SETS["combined"], list(BASE_NAMES) + list(CUSTOM))


if __name__ == "__main__":
    unittest.main()
