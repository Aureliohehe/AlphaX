import unittest

import numpy as np
import pandas as pd

from csi300_data import ticker_features


class CSIAdapterTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2020-01-01", periods=180)
        close = np.arange(180, dtype=float) + 100
        self.raw = pd.DataFrame({"date": self.calendar, "ticker": "SH600000", "close": close,
                                 "open": close - .2, "high": close + 1, "low": close - 1, "volume": 1000.})
        self.intervals = pd.DataFrame({"start": [self.calendar[140]], "end": [self.calendar[170]]})
        self.columns = ["date", "ticker", "return_1d", "return_5d", "sma_120_ratio", "alpha158_cntd20",
                        "target_1d", "target_5d", "label_end_1d", "label_end_5d"]

    def build(self, raw=None):
        return ticker_features(self.raw if raw is None else raw, self.calendar, self.intervals,
                               self.columns, self.calendar[0], self.calendar[-1]).set_index("date")

    def test_warmup_before_entry_and_labels_after_exit(self):
        result = self.build()
        self.assertEqual(len(result), 31)
        expected = 240 / np.mean(np.arange(121, 241)) - 1
        self.assertAlmostEqual(result.loc[self.calendar[140], "sma_120_ratio"], expected)
        self.assertAlmostEqual(result.loc[self.calendar[170], "target_5d"], 275 / 270 - 1)
        self.assertEqual(result.loc[self.calendar[170], "label_end_5d"], self.calendar[175])

    def test_missing_session_is_not_filled_or_removed(self):
        raw = self.raw.drop(index=150)
        result = self.build(raw)
        self.assertIn(self.calendar[150], result.index)
        self.assertTrue(pd.isna(result.loc[self.calendar[150], "return_1d"]))
        self.assertTrue(pd.isna(result.loc[self.calendar[151], "return_1d"]))
        self.assertTrue(pd.isna(result.loc[self.calendar[149], "target_1d"]))
        self.assertEqual(result.loc[self.calendar[149], "label_end_1d"], self.calendar[150])
        self.assertTrue(pd.isna(result.loc[self.calendar[155], "alpha158_cntd20"]))

    def test_future_prices_do_not_change_past_features(self):
        before = self.build()
        raw = self.raw.copy()
        raw.loc[160:, ["open", "high", "low", "close"]] *= 3
        after = self.build(raw)
        features = ["return_1d", "return_5d", "sma_120_ratio", "alpha158_cntd20"]
        pd.testing.assert_frame_equal(before.loc[:self.calendar[159], features], after.loc[:self.calendar[159], features])


if __name__ == "__main__":
    unittest.main()
