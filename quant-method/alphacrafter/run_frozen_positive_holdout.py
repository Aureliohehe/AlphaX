"""Freeze one development-selected research portfolio, then evaluate its holdout once.

This is an explicit research return test, not an override of formal admission.
No holdout return is used to select factors, parameters, or a winning strategy.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cn_portfolio import CNConfig, simulate_cn
from medium_frequency import factor_panel, load_member_bars, screen_signals
from run_cn_portfolio_research import load_quotes
from run_medium_attribution import trial_metrics
from run_signal_stability import static_signals, verify_inputs
from trader_lite import file_fingerprint

HERE = Path(__file__).resolve().parent
ML = HERE.parent/'machine-learn'


def choose_development_configuration(candidates):
    """Maximize worst calendar-year net return across both development costs.

    Both development cumulative returns must be positive. Ties favor the
    higher worst cumulative return, then original order. Never read holdout.
    Formal portfolio gates remain separate and do not become admission here.
    """
    eligible = []
    for index, row in enumerate(candidates):
        trials = row['development_trials']
        if {r['cost_multiplier'] for r in trials} != {1, 2} or len(trials) != 2:
            raise ValueError('Require exactly two development cost scenarios')
        if any(r['start'] != '2022-01-04' or r['end'] != '2023-12-29' for r in trials):
            raise ValueError('Selection may read development windows only')
        values = []
        for trial in trials:
            if set(trial['calendar_year_returns']) != {'2022', '2023'}:
                raise ValueError('Unexpected development years')
            numbers = [trial['cumulative_return'], *trial['calendar_year_returns'].values()]
            if not all(isinstance(n, (float, int)) and np.isfinite(n) for n in numbers):
                raise ValueError('Nonfinite development returns')
            audit = trial['execution_audit']
            if not all(audit[k] for k in ('liquidation_complete', 'turnover_budget_respected', 'cost_budget_respected')):
                raise ValueError('Failed development execution audit')
            values += list(trial['calendar_year_returns'].values())
        worst_cumulative = min(t['cumulative_return'] for t in trials)
        if worst_cumulative > 0:
            eligible.append((min(values), worst_cumulative, -index, row))
    if not eligible:
        raise ValueError('No positive development research configuration')
    return max(eligible, key=lambda r: r[:3])[3]


def verify_completed_experiment(root):
    read = lambda name: json.loads((root/name).read_text())
    if read('status.json')['status'] != 'completed_diagnostic':
        raise ValueError('Require a completed development experiment')
    frozen, summary = read('frozen.json'), read('summary.json')
    if frozen['window'] != ['2022-01-01', '2023-12-31'] or summary['holdout_evaluated']:
        raise ValueError('Expected development-only inputs')
    for name, expected in frozen['inputs'].items():
        path = Path(name)
        checked = root/'source_snapshot'/path.name if path.suffix == '.py' else path
        actual = file_fingerprint(checked)
        if any(actual[k] != expected[k] for k in ('sha256', 'bytes')):
            raise ValueError('Development source changed: '+name)
    audit = read('artifact_audit.json')
    for result in audit['trials']:
        if not result['passed']:
            raise ValueError('Failed development ledger audit')
        for name, expected in result['artifacts'].items():
            if file_fingerprint(root/name) != expected:
                raise ValueError('Development artifact changed: '+name)
    return frozen, summary


def audit_ledger(report, trades, positions, calendar, config):
    """Independently reconstruct cash, inventory, costs, NAV and turnover."""
    previous_nav = report.nav.shift(1, fill_value=1.)
    np.testing.assert_allclose((1+report.net_return).cumprod(), report.nav, rtol=1e-11, atol=1e-12)
    equity = (positions.adjusted_units*positions.mark).groupby(positions.date).sum().reindex(report.index, fill_value=0.)
    np.testing.assert_allclose(equity+report.cash, report.nav, atol=1e-12)
    fees = trades.groupby('date').total_cost.sum().reindex(report.index, fill_value=0.)
    notional = trades.groupby('date').notional_nav_units.sum().reindex(report.index, fill_value=0.)
    np.testing.assert_allclose(fees/previous_nav, report.transaction_cost, atol=1e-12)
    np.testing.assert_allclose(notional/previous_nav/2, report.turnover, atol=1e-12)
    np.testing.assert_allclose(trades.total_cost, trades[['commission', 'slippage', 'stamp_duty']].sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(trades.commission, trades.notional_nav_units*config.commission, atol=1e-12)
    np.testing.assert_allclose(trades.slippage, trades.notional_nav_units*config.slippage, atol=1e-12)
    tax_rate = np.where(trades.date >= pd.Timestamp('2023-08-28'), .0005, .001)
    np.testing.assert_allclose(trades.stamp_duty, trades.notional_nav_units*tax_rate*(trades.side == 'sell'), atol=1e-12)
    if (report.cash < -1e-10).any() or (report.turnover > config.max_daily_turnover+1e-10).any() or (report.transaction_cost > config.max_daily_cost+1e-10).any():
        raise ValueError('Cash or budget violation')
    cash, inventory, bought = 1., {}, {}
    for day in report.index:
        for trade in trades.loc[trades.date == day].itertuples():
            if trade.signal_date != calendar[calendar.get_loc(day)-1]:
                raise ValueError('Noncausal trade signal')
            if trade.side == 'buy':
                cash -= trade.notional_nav_units+trade.total_cost
                inventory[trade.ticker] = inventory.get(trade.ticker, 0.)+trade.adjusted_units
                bought[trade.ticker] = day
            else:
                if trade.ticker not in bought or calendar.get_loc(day)-calendar.get_loc(bought[trade.ticker]) < config.min_holding_sessions:
                    raise ValueError('Holding lock violation')
                if not (report.loc[day, 'scheduled_rebalance'] or report.loc[day, 'terminal_winddown']):
                    raise ValueError('Off-schedule sell')
                inventory[trade.ticker] -= trade.adjusted_units
                if inventory[trade.ticker] < -1e-10:
                    raise ValueError('Unowned inventory sold')
                cash += trade.notional_nav_units-trade.total_cost
        if abs(cash-report.loc[day, 'cash']) > 1e-10:
            raise ValueError('Cash ledger mismatch')
        recorded = positions.loc[positions.date == day].set_index('ticker').adjusted_units.to_dict()
        if any(abs(inventory.get(t, 0.)-recorded.get(t, 0.)) > 1e-10 for t in inventory.keys() | recorded.keys()):
            raise ValueError('Position ledger mismatch')
    if any(abs(q) > 1e-10 for q in inventory.values()):
        raise ValueError('Incomplete liquidation')
    return {'passed': True, 'trades': len(trades), 'sessions': len(report), 'ending_inventory': 0,
            'checks': ['NAV_compounding', 'positions_NAV', 'cash_inventory_ledger', 'cost_components',
                       'historical_tax', 'budgets', 'turnover', 'previous_session_signal', 'holding_locks', 'sell_schedule']}


def verify_frozen_inputs(out, frozen):
    for name, expected in frozen['inputs'].items():
        path = Path(name)
        # The runner alone may be amended for an explicitly documented repair;
        # its original immutable snapshot still has to match the first freeze.
        checked = out/'source_snapshot'/path.name if path == Path(__file__).resolve() else path
        actual = file_fingerprint(checked)
        if any(actual[k] != expected[k] for k in ('sha256', 'bytes')):
            raise ValueError('Frozen input changed: '+name)


def check_feature_prefix(original, prefix, fields):
    # read_csv(usecols=...) preserves file order, not the usecols list order.
    columns = ['date', 'ticker', *fields]
    # HDF arithmetic retains float32 for simple ratios; CSV parsing promotes
    # their shortest decimal strings to float64. Restore each original source
    # dtype before comparison instead of treating serialization as a change.
    restored = original[columns].copy()
    for field in fields:
        restored[field] = restored[field].astype(prefix[field].dtype)
    pd.testing.assert_frame_equal(restored, prefix[columns], rtol=1e-11, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-prefix-order-failure', action='store_true',
                        help='Resume the original frozen selection after the known pre-return column-order assertion failure')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.resume_prefix_order_failure:
        frozen = json.loads((out/'selection_frozen.json').read_text())
        failure = json.loads((out/'status.json').read_text())
        serialization_failure = ('DataFrame.columns are different' in failure.get('error', '') or
                                 'Attribute "dtype" are different' in failure.get('error', ''))
        if failure['status'] != 'failed_holdout_consumed' or not serialization_failure:
            parser.error('Recovery only permits known prefix serialization assertions, before return evaluation')
        if any(out.glob('cost*_returns.csv')):
            parser.error('Return evaluation already exists')
        verify_frozen_inputs(out, frozen)
        recovery = {'created_at': datetime.now(timezone.utc).isoformat(), 'previous_failure': failure,
                    'original_selection': file_fingerprint(out/'selection_frozen.json'),
                    'runner': file_fingerprint(Path(__file__).resolve()),
                    'change': 'Align named columns and restore CSV features to original source precision for the prefix check; freeze/selection/rules/data/costs unchanged',
                    'selection_rerun': False, 'holdout_returns_previously_computed': False}
        attempt = 1+len(list(out.glob('recovery_frozen*.json')))
        record = out/('recovery_frozen.json' if attempt == 1 else f'recovery_frozen_{attempt}.json')
        record.write_text(json.dumps(recovery, indent=2)+'\n')
        shutil.copyfile(__file__, out/f'source_snapshot/repaired{attempt}_run_frozen_positive_holdout.py')
        evaluate_frozen(out, frozen)
        return
    if out.exists():
        parser.error('Use a new directory; completed/failed holdout runs are never overwritten')
    research = ML/'data/qlib/alphacrafter_medium_frequency_20261117'
    attribution = ML/'data/qlib/alphacrafter_medium_attribution_20261117'
    development = [ML/'data/qlib'/name for name in ('alphacrafter_signal_stability_20261117_v2',
                                                   'alphacrafter_absolute_trend_20261117')]
    verify_inputs(research, attribution)
    candidates, evidence = [], []
    for root in development:
        frozen, summary = verify_completed_experiment(root)
        for variant in frozen['variants']:
            trials = [t for t in summary['trials'] if t['variant'] == variant['name']]
            # Recompute selection inputs from actual development return paths.
            for trial in trials:
                path = root/f"{variant['name']}_cost{trial['cost_multiplier']}_returns.csv"
                returns = pd.read_csv(path, parse_dates=['date'])
                years = {str(y): float((1+g.net_return).prod()-1) for y, g in returns.groupby(returns.date.dt.year)}
                np.testing.assert_allclose([years[y] for y in ('2022', '2023')],
                    [trial['calendar_year_returns'][y] for y in ('2022', '2023')], rtol=1e-11, atol=1e-12)
                if abs(float((1+returns.net_return).prod()-1)-trial['cumulative_return']) > 1e-11:
                    raise ValueError('Development summary does not reconcile')
                evidence.append(path)
            candidates.append({'source': str(root), 'variant': variant, 'development_trials': trials})
        evidence += [root/n for n in ('frozen.json', 'summary.json', 'artifact_audit.json', 'status.json')]
    selected = choose_development_configuration(candidates)
    variant = selected['variant']
    standard = next(t for t in selected['development_trials'] if t['cost_multiplier'] == 1)
    config = CNConfig(**standard['execution_audit']['config'])
    raw = ML/'data/raw/qlib_csi300_quantaalpha'
    paths = evidence+[raw/n for n in ('daily_pv.h5', 'qlib_day.txt', 'qlib_csi300.txt')]
    paths += [research/'features_labels.csv.gz', attribution/'experiment_frozen.json']
    paths += [HERE/n for n in ('run_frozen_positive_holdout.py', 'cn_portfolio.py', 'medium_frequency.py',
                              'run_signal_stability.py', 'run_medium_attribution.py', 'run_cn_portfolio_research.py', 'trader_lite.py')]
    hashes = {str(p): file_fingerprint(p) for p in paths}
    out.mkdir(parents=True)
    def write(name, data):
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    now = lambda: datetime.now(timezone.utc).isoformat()
    frozen = {'artifact_type': 'frozen_positive_return_research_holdout_v1', 'created_at': now(),
              'objective': 'Positive cumulative net return on the full reserved period; cost stress reported separately',
              'scope': 'research_simulation_not_formal_strategy_admission', 'formal_strategy_admitted': False,
              'formal_gate_development_passed': standard['passes_numeric_strategy_thresholds'],
              'selection_rule': 'Among variants with positive development cumulative net return at both costs, maximize worst development calendar-year net return across both costs; ties higher worst cumulative return, then original order',
              'selection_is_post_development_research': True, 'candidate_count': len(candidates),
              'selected': selected, 'holdout': ['2024-01-01', '2025-12-31'],
              'phase': 'Reset fixed 5-session rebalance phase at first holdout execution session; start with cash; budgeted terminal liquidation',
              'cost_scenarios': {str(m): asdict(replace(config, commission=m*config.commission, slippage=m*config.slippage)) for m in (1, 2)},
              'inputs': hashes, 'holdout_policy': 'One selected signal rule; no retuning after observing holdout; both costs predeclared',
              'limitations': ['Historical member/calendar/adjustment data not independently PIT certified',
                              'Adjusted fractional units; approximate gap filter; no lots, minimum commission, depth or exact ST/limit/suspension reconstruction',
                              'Development experiments bypassed factor admission and failed the formal portfolio gate; this run does not change that gate',
                              'Historical holdout of this research branch, not live or prospective performance']}
    write('selection_frozen.json', frozen)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for p in paths:
        if p.suffix == '.py':
            shutil.copyfile(p, snapshot/p.name)
    print(json.dumps({'selected': variant, 'config': asdict(config), 'frozen_before_holdout': now()}), flush=True)
    evaluate_frozen(out, frozen)


def evaluate_frozen(out, frozen):
    def write(name, data):
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    now = lambda: datetime.now(timezone.utc).isoformat()
    runtime_hash = file_fingerprint(Path(__file__).resolve())
    verify_frozen_inputs(out, frozen)
    raw = ML/'data/raw/qlib_csi300_quantaalpha'
    research = ML/'data/qlib/alphacrafter_medium_frequency_20261117'
    attribution = ML/'data/qlib/alphacrafter_medium_attribution_20261117'
    variant = frozen['selected']['variant']
    write('status.json', {'status': 'holdout_started', 'holdout_first_read_at': now()})
    try:
        calendar = pd.DatetimeIndex(pd.to_datetime((raw/'qlib_day.txt').read_text().splitlines()))
        start, end = frozen['holdout']
        first = calendar[calendar >= start][0]
        # Prepare all historical members before filtering; rolling features use
        # each stock's full available past, not only its membership interval.
        bars, intervals = load_member_bars(raw/'daily_pv.h5', raw/'qlib_csi300.txt', '2023-01-01', end)
        fields = sorted({f for group in variant.get('groups', []) for f, _ in group})
        if not fields:
            prior = json.loads((attribution/'experiment_frozen.json').read_text())
            fields = [e['candidate_id'] for e in prior['pool']['candidates_by_horizon']['5']]
        pieces = []
        for i, (ticker, stock) in enumerate(bars.groupby('ticker', sort=True), 1):
            values = factor_panel(stock, calendar, end=end)
            keep = pd.Series(False, index=values.index)
            for interval in intervals.loc[intervals.ticker == ticker].itertuples():
                keep |= values.date.between(interval.start, interval.end)
            keep &= values.date.between('2023-01-01', end)
            columns = ['date', 'ticker', *fields]
            if 'screening' in variant:
                columns += ['target_open_open_5d', 'label_end_5d']
            pieces.append(values.loc[keep, columns])
            if i % 100 == 0:
                print(f'Prepared {i} holdout stock histories', flush=True)
        frame = pd.concat(pieces, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)
        del bars, pieces
        if frame.duplicated(['date', 'ticker']).any() or frame.date.max() > pd.Timestamp(end):
            raise ValueError('Invalid holdout feature dates/keys')
        # Exact 2023 feature prefix must survive extension into the holdout.
        original = pd.read_csv(research/'features_labels.csv.gz', usecols=['date', 'ticker', *fields], parse_dates=['date'])
        original = original.loc[original.date >= '2023-01-01'].sort_values(['date', 'ticker']).reset_index(drop=True)
        prefix = frame.loc[frame.date <= '2023-12-31', ['date', 'ticker', *fields]].reset_index(drop=True)
        check_feature_prefix(original, prefix, fields)
        write('feature_audit.json', {'rows': len(frame), 'min_date': str(frame.date.min().date()),
              'max_date': str(frame.date.max().date()), 'prefix_2023_rows': len(prefix), 'prefix_matches_development': True,
              'fields': fields, 'future_labels_used_for_static_signals': False})
        frame.to_csv(out/'features.csv.gz', index=False)
        if 'groups' in variant:
            signals = static_signals(frame, variant['groups'], calendar, first, start, end,
                                     positive_features=variant.get('positive_features', ()))
        else:
            pool = deepcopy(prior['pool'])
            wanted = {'mf_low_volatility_60', 'mf_low_downside_risk_60', 'mf_momentum_120_skip20',
                      'mf_trend_60_120', 'mf_volume_stability_60'}
            pool['candidates_by_horizon']['5'] = [e for e in pool['candidates_by_horizon']['5'] if e['candidate_id'] in wanted]
            signals, selections = screen_signals(frame, pool, 5, calendar, first, start, end, **variant['screening'])
            selections.to_csv(out/'screening.csv', index=False)
        signals.to_csv(out/'signals.csv.gz', index=False)
        quotes = load_quotes(raw/'daily_pv.h5', set(frame.ticker), calendar, start, end)
        results, audits = [], []
        for m, settings in frozen['cost_scenarios'].items():
            cfg = CNConfig(**settings)
            report, trades, positions, execution_audit = simulate_cn(signals, quotes, calendar, cfg, start, end)
            audit = audit_ledger(report, trades, positions, calendar, cfg)
            trial = {'cost_multiplier': int(m), **trial_metrics(report, trades, execution_audit)}
            results.append(trial)
            audits.append({'cost_multiplier': int(m), **audit})
            report.to_csv(out/f'cost{m}_returns.csv')
            trades.to_csv(out/f'cost{m}_trades.csv', index=False)
            positions.to_csv(out/f'cost{m}_positions.csv.gz', index=False)
            print(json.dumps({k: trial[k] for k in ('cost_multiplier', 'cumulative_return', 'annualized_return',
                'sharpe', 'max_drawdown', 'calendar_year_returns', 'passes_numeric_strategy_thresholds')}), flush=True)
        verify_frozen_inputs(out, frozen)
        if file_fingerprint(Path(__file__).resolve()) != runtime_hash:
            raise ValueError('Runner changed during evaluation')
        # status.json is updated after finalization and is not immutable here.
        artifacts = {p.name: file_fingerprint(p) for p in out.iterdir() if p.is_file() and p.name != 'status.json'}
        write('artifact_audit.json', {'audits': audits, 'artifacts': artifacts})
        standard = next(t for t in results if t['cost_multiplier'] == 1)
        write('summary.json', {'scope': frozen['scope'], 'holdout_evaluated': True, 'selected_variant': variant,
              'formal_strategy_admitted': False, 'formal_development_gate_passed': frozen['formal_gate_development_passed'],
              'positive_net_return_observed': standard['cumulative_return'] > 0,
              'positive_at_both_costs': all(t['cumulative_return'] > 0 for t in results), 'trials': results})
        write('status.json', {'status': 'completed_research_holdout', 'completed_at': now()})
    except BaseException as exc:
        write('status.json', {'status': 'failed_holdout_consumed', 'error': str(exc), 'failed_at': now()})
        raise


if __name__ == '__main__':
    main()
