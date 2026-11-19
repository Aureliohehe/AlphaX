from copy import deepcopy
from dataclasses import replace
from io import StringIO
import unittest

import pandas as pd
import numpy as np

from cn_portfolio import simulate_cn
from medium_frequency import strategy_presets
from run_frozen_positive_holdout import audit_ledger, check_feature_prefix, choose_development_configuration


def candidate(name, years, cumulative):
    trials = [{'cost_multiplier': m, 'start': '2022-01-04', 'end': '2023-12-29',
               'calendar_year_returns': {'2022': years[0], '2023': years[1]},
               'cumulative_return': cumulative,
               'execution_audit': {'liquidation_complete': True, 'turnover_budget_respected': True,
                                   'cost_budget_respected': True}} for m in (1, 2)]
    return {'variant': {'name': name}, 'development_trials': trials}


class FrozenHoldoutTests(unittest.TestCase):
    def test_prefix_check_aligns_column_order_but_rejects_feature_changes(self):
        first = pd.DataFrame({'date': [pd.Timestamp('2023-01-01')], 'ticker': ['A'], 'z': [.1], 'a': [.2]})
        second = first[['date', 'ticker', 'a', 'z']].copy()
        check_feature_prefix(first, second, ['a', 'z'])
        second['z'] = .3
        with self.assertRaises(AssertionError):
            check_feature_prefix(first, second, ['a', 'z'])
        single_precision = first.copy()
        single_precision['z'] = np.float32(.123456789)
        parsed = pd.read_csv(StringIO(single_precision.to_csv(index=False)), parse_dates=['date'])
        check_feature_prefix(parsed, single_precision, ['a', 'z'])

    def test_selection_prefers_worst_year_not_largest_total_and_rejects_future(self):
        stable = candidate('stable', (-.02, .08), .0584)
        high_return = candidate('high', (-.12, .50), .32)
        self.assertEqual(choose_development_configuration([high_return, stable])['variant']['name'], 'stable')
        changed = deepcopy(stable)
        changed['development_trials'][0]['end'] = '2025-12-31'
        with self.assertRaisesRegex(ValueError, 'development windows'):
            choose_development_configuration([changed])
        changed = deepcopy(stable)
        changed['development_trials'][0]['calendar_year_returns']['2024'] = 1.
        with self.assertRaisesRegex(ValueError, 'development years'):
            choose_development_configuration([changed])

    def test_selection_requires_both_costs_positive_and_valid_audits(self):
        value = candidate('x', (-.02, .08), .05)
        value['development_trials'][1]['cumulative_return'] = -.01
        with self.assertRaisesRegex(ValueError, 'No positive'):
            choose_development_configuration([value])
        value = candidate('x', (-.02, .08), .05)
        value['development_trials'][1]['calendar_year_returns']['2022'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            choose_development_configuration([value])
        value = candidate('x', (-.02, .08), .05)
        value['development_trials'][1]['execution_audit']['liquidation_complete'] = False
        with self.assertRaisesRegex(ValueError, 'execution audit'):
            choose_development_configuration([value])

    def test_independent_ledger_audit_detects_corrupted_cash_and_future_signals(self):
        calendar = pd.bdate_range('2024-01-01', periods=60)
        quotes = pd.DataFrame([{'date': day, 'ticker': ticker, 'open': 100+i*.1, 'close': 100+i*.1}
                               for i, day in enumerate(calendar) for ticker in ('A', 'SH000300')])
        signals = pd.DataFrame({'date': calendar, 'ticker': 'A', 'prediction': 1.})
        config = replace(strategy_presets()[5], topk=1, fill_target_daily=True)
        report, trades, positions, _ = simulate_cn(signals, quotes, calendar, config, calendar[1], calendar[-1])
        self.assertTrue(audit_ledger(report, trades, positions, calendar, config)['passed'])
        corrupted = report.copy()
        corrupted.loc[calendar[10], 'cash'] += .01
        with self.assertRaises(AssertionError):
            audit_ledger(corrupted, trades, positions, calendar, config)
        corrupted_trades = trades.copy()
        corrupted_trades.loc[0, 'signal_date'] = corrupted_trades.loc[0, 'date']
        with self.assertRaisesRegex(ValueError, 'Noncausal'):
            audit_ledger(report, corrupted_trades, positions, calendar, config)


if __name__ == '__main__':
    unittest.main()
