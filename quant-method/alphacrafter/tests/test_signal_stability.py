import unittest
from dataclasses import replace
from io import StringIO

import numpy as np
import pandas as pd

from run_signal_stability import static_signals
from cn_portfolio import simulate_cn
from medium_frequency import strategy_presets


class StaticSignalCausalityTests(unittest.TestCase):
    def test_near_ties_and_csv_roundtrip_preserve_trades(self):
        calendar = pd.bdate_range('2022-01-03', periods=50)
        quotes = pd.DataFrame([{'date': d, 'ticker': t, 'open': 100.+i*(t == 'A'),
                                'close': 100.+i*(t == 'A')}
                               for i, d in enumerate(calendar) for t in ('A', 'B', 'SH000300')])
        signals = pd.DataFrame([{'date': d, 'ticker': t, 'prediction': .8+1e-16*(t == 'B')}
                                for d in calendar for t in ('B', 'A')])
        config = replace(strategy_presets()[5], topk=1, fill_target_daily=True)
        args = (quotes, calendar, config, calendar[1], calendar[-1])
        report, trades, _, _ = simulate_cn(signals, *args)
        changed = pd.read_csv(StringIO(signals.to_csv(index=False)), parse_dates=['date'])
        changed['prediction'] = .8+1e-16*(changed.ticker == 'A')
        other, other_trades, _, _ = simulate_cn(changed, *args)
        self.assertEqual(set(trades.ticker), {'A'})
        pd.testing.assert_frame_equal(report, other)
        pd.testing.assert_frame_equal(trades, other_trades)

    def test_future_features_and_all_labels_do_not_change_past_signals(self):
        calendar = pd.bdate_range('2022-01-03', periods=45)
        frame = pd.DataFrame([{'date': d, 'ticker': str(t), 'a': t+i*.1,
                               'b': (t+2*i)%5, 'target_open_open_5d': t}
                              for i, d in enumerate(calendar) for t in range(5)])
        args = ([[('a', 1)], [('b', -1)]], calendar, calendar[1], calendar[1], calendar[-1])
        before = static_signals(frame, *args)
        modified = frame.copy()
        cutoff = calendar[20]
        modified.loc[modified.date > cutoff, ['a', 'b']] *= -100
        modified['target_open_open_5d'] *= -1000
        after = static_signals(modified, *args)
        pd.testing.assert_frame_equal(before[before.date <= cutoff], after[after.date <= cutoff])
        expected = set(calendar[::5])
        self.assertEqual(set(before.date), expected)

    def test_incomplete_components_cannot_become_high_ranked_signals(self):
        calendar = pd.bdate_range('2022-01-03', periods=10)
        frame = pd.DataFrame({'date': [calendar[0]]*3, 'ticker': ['A', 'B', 'C'],
                              'a': [3., 2., 1.], 'b': [np.nan, 1., 2.]})
        result = static_signals(frame, [[('a', 1)], [('b', -1)]], calendar,
                                calendar[1], calendar[1], calendar[-1]).set_index('ticker')
        self.assertTrue(pd.isna(result.loc['A', 'prediction']))
        self.assertGreater(result.loc['B', 'prediction'], result.loc['C', 'prediction'])

    def test_absolute_trend_filter_excludes_negative_and_unknown_trends(self):
        calendar = pd.bdate_range('2022-01-03', periods=10)
        frame = pd.DataFrame({'date': [calendar[0]]*4, 'ticker': ['A', 'B', 'C', 'D'],
                              'a': [4., 3., 2., 1.], 'trend': [.1, -.1, 0., np.nan]})
        args = ([[('a', 1)]], calendar, calendar[1], calendar[1], calendar[-1])
        result = static_signals(frame, *args, positive_features=['trend']).set_index('ticker')
        self.assertEqual(set(result.prediction.dropna().index), {'A'})
        frame.loc[frame.ticker == 'A', 'trend'] = -.1
        self.assertTrue(static_signals(frame, *args, positive_features=['trend']).prediction.isna().all())


if __name__ == '__main__':
    unittest.main()
