"""Bounded development experiment: stable defensive/trend composites versus IC switching."""
import argparse
import ast
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cn_portfolio import simulate_cn
from medium_frequency import anchor_mask, screen_signals, strategy_presets
from run_cn_portfolio_research import load_quotes
from run_medium_attribution import DEFENSIVE, TREND, trial_metrics
from trader_lite import file_fingerprint

HERE = Path(__file__).resolve().parent


def verify_inputs(src, attribution):
    """Verify historical inputs and the actual functions used by this replay.

    Unused Miner/1d composite code has changed since the original experiment.
    Preserve its original snapshots, without requiring unrelated live code to
    match. The medium-frequency signal and execution dependencies must match.
    """
    manifest = json.loads((src/'manifest.json').read_text())
    if json.loads((src/'status.json').read_text())['status'] != 'completed_research':
        raise ValueError('Incomplete source research')
    if manifest['visible_end'] != '2023-12-31':
        raise ValueError('Unexpected source window')
    for name, expected in manifest['sources'].items():
        path = Path(name)
        checked = src/'source_snapshot'/path.name if path.suffix == '.py' else path
        actual = file_fingerprint(checked)
        if any(actual[k] != expected[k] for k in ('sha256', 'bytes')):
            raise ValueError('Historical source changed: '+name)
    prior = json.loads((attribution/'experiment_frozen.json').read_text())
    for name, expected in prior['input_hashes'].items():
        path = Path(name)
        if path.suffix != '.py' and file_fingerprint(path) != expected:
            raise ValueError('Attribution input changed: '+name)
        if path.suffix == '.py':
            actual = file_fingerprint(attribution/'source_snapshot'/path.name)
            if any(actual[k] != expected[k] for k in ('sha256', 'bytes')):
                raise ValueError('Attribution snapshot changed: '+name)
    for name in ('medium_frequency.py', 'trader_lite.py'):
        if (HERE/name).read_bytes() != (attribution/'source_snapshot'/name).read_bytes():
            raise ValueError('Replay dependency changed: '+name)
    for filename, function in (('run_cn_portfolio_research.py', 'load_quotes'),
                               ('run_medium_attribution.py', 'trial_metrics')):
        def code(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body
                                 if isinstance(n, ast.FunctionDef) and n.name == function))
        if code(HERE/filename) != code(attribution/'source_snapshot'/filename):
            raise ValueError('Replay function changed: '+function)


def static_signals(frame, groups, calendar, first_execution, start, end, positive_features=()):
    """Equal weight within groups, then across groups; require every component.

    Group members are (feature, frozen direction) pairs. No labels are read.
    Only scheduled T-close observations are emitted for T+1 execution.
    """
    if not groups or any(not group for group in groups):
        raise ValueError('Empty factor group')
    if frame.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate feature keys')
    first = calendar[calendar >= pd.Timestamp(start)][0]
    previous = calendar[calendar.get_loc(first)-1]
    wanted = anchor_mask(frame.date, calendar, first_execution, 5) & frame.date.between(previous, end)
    visible = frame.loc[wanted].copy()
    scores = []
    for group in groups:
        members = []
        for feature, direction in group:
            if direction not in (-1, 1):
                raise ValueError('Invalid factor direction')
            values = (direction*visible[feature]).replace([np.inf, -np.inf], np.nan)
            members.append(values.groupby(visible.date).rank(pct=True))
        scores.append(pd.concat(members, axis=1).sum(axis=1, min_count=len(members))/len(members))
    visible['prediction'] = pd.concat(scores, axis=1).sum(axis=1, min_count=len(scores))/len(scores)
    # An absolute-trend filter uses only the current close's known features.
    # Ineligible stocks are absent from the ranking, including retention ranks.
    if positive_features:
        eligible = visible[list(positive_features)].gt(0).all(axis=1)
        visible.loc[~eligible, 'prediction'] = np.nan
    return visible[['date', 'ticker', 'prediction']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-dir', type=Path, required=True)
    parser.add_argument('--attribution-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment', choices=('stability', 'absolute_trend'), default='stability')
    args = parser.parse_args()
    src, attribution, out = (p.resolve() for p in (args.research_dir, args.attribution_dir, args.output_dir))
    if out.exists():
        parser.error('Use a new output directory')
    verify_inputs(src, attribution)
    if json.loads((attribution/'status.json').read_text())['status'] != 'completed_diagnostic':
        raise ValueError('Incomplete attribution source')
    prior = json.loads((attribution/'experiment_frozen.json').read_text())
    pool = deepcopy(prior['pool'])
    names = DEFENSIVE+TREND+['mf_volume_stability_60']
    entries = {e['candidate_id']: e for e in pool['candidates_by_horizon']['5']}
    pool['candidates_by_horizon']['5'] = [entries[n] for n in names]
    pairs = lambda members: [(n, entries[n]['direction']) for n in members]
    variants = [
        {'name': 'dynamic_12', 'screening': {'lookback_periods': 12, 'min_periods': 6}},
        {'name': 'dynamic_52', 'screening': {'lookback_periods': 52, 'min_periods': 26}},
        {'name': 'static_equal5', 'groups': [pairs(names)]},
        {'name': 'static_balanced4', 'groups': [pairs(DEFENSIVE), pairs(TREND)]},
        {'name': 'static_defensive', 'groups': [pairs(DEFENSIVE)]},
        {'name': 'static_trend', 'groups': [pairs(TREND)]},
    ]
    if args.experiment == 'absolute_trend':
        variants = [
            {'name': 'balanced30', 'groups': [pairs(DEFENSIVE), pairs(TREND)], 'topk': 30},
            {'name': 'positive_trend30', 'groups': [pairs(DEFENSIVE), pairs(TREND)],
             'positive_features': TREND, 'topk': 30},
            {'name': 'positive_trend15', 'groups': [pairs(DEFENSIVE), pairs(TREND)],
             'positive_features': TREND, 'topk': 15},
        ]
    base = replace(strategy_presets()[5], fill_target_daily=True)
    configs = {str(m): replace(base, commission=m*base.commission, slippage=m*base.slippage) for m in (1, 2)}
    raw = HERE.parent/'machine-learn/data/raw/qlib_csi300_quantaalpha'
    paths = [src/n for n in ('features_labels.csv.gz', 'manifest.json', 'research_assessments.json')]
    paths += [attribution/n for n in ('experiment_frozen.json', '5d_fill_without_reversal_returns.csv',
                                    '5d_fill_without_reversal_signals.csv.gz')]
    paths += [raw/n for n in ('daily_pv.h5', 'qlib_day.txt')]
    paths += [HERE/n for n in ('run_signal_stability.py', 'cn_portfolio.py', 'medium_frequency.py',
                              'run_medium_attribution.py', 'run_cn_portfolio_research.py', 'trader_lite.py')]
    hashes = {str(p): file_fingerprint(p) for p in paths}
    out.mkdir(parents=True)
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    write('frozen.json', {'scope': 'development_only_admission_bypassed', 'holdout_evaluated': False,
          'experiment': args.experiment,
          'window': ['2022-01-01', '2023-12-31'], 'variants': variants,
          'configs': {m: asdict(c) for m, c in configs.items()}, 'inputs': hashes,
          'hypothesis': ('Short IC switching may dilute persistent defensive/long trend scores; compare longer feedback and fixed composites'
                         if args.experiment == 'stability' else
                         'Relative ranks can buy absolute downtrends; require both long trend features positive; test 30/15-stock diversification'),
          'ranking': 'Scores rounded to 12 decimal places; ties resolved by ticker; original signals replayed with this same corrected executor',
          'selection': 'No automatic strategy admission; all variants reported; development window previously examined'})
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for path in paths:
        if path.suffix == '.py':
            shutil.copyfile(path, snapshot/path.name)
    write('status.json', {'status': 'running'})
    try:
        frame = pd.read_csv(src/'features_labels.csv.gz')
        for col in frame:
            if col == 'date' or col.startswith(('label_start_', 'label_end_')):
                frame[col] = pd.to_datetime(frame[col])
        if frame.date.max() > pd.Timestamp('2023-12-31'):
            raise ValueError('Reserved-period features')
        calendar = pd.DatetimeIndex(pd.to_datetime((raw/'qlib_day.txt').read_text().splitlines()))
        first = calendar[calendar >= '2022-01-01'][0]
        quotes = load_quotes(raw/'daily_pv.h5', set(frame.ticker), calendar, '2022-01-01', '2023-12-31')
        trials = []
        for variant in variants:
            name = variant['name']
            print('Signals: '+name, flush=True)
            if 'screening' in variant:
                signals, selections = screen_signals(frame, pool, 5, calendar, first, '2022-01-01', '2023-12-31', **variant['screening'])
                selections.to_csv(out/f'{name}_screening.csv', index=False)
                used = pd.to_datetime(selections.last_label_end_used)
                if not (used.isna() | (used <= selections.date)).all():
                    raise ValueError('Unmatured labels')
            else:
                signals = static_signals(frame, variant['groups'], calendar, first, '2022-01-01', '2023-12-31',
                                         positive_features=variant.get('positive_features', ()))
            signals.to_csv(out/f'{name}_signals.csv.gz', index=False)
            for m, config in configs.items():
                if 'topk' in variant:
                    config = replace(config, topk=variant['topk'], sell_rank_buffer=variant['topk'])
                report, trades, positions, audit = simulate_cn(signals, quotes, calendar, config, '2022-01-01', '2023-12-31')
                if name == 'dynamic_12' and m == '1':
                    # Replay original saved signals with the same canonical
                    # score/ticker ordering, rather than compare old ULP ties.
                    original_signals = pd.read_csv(attribution/'5d_fill_without_reversal_signals.csv.gz', parse_dates=['date'])
                    old, _, _, _ = simulate_cn(original_signals, quotes, calendar, config, '2022-01-01', '2023-12-31')
                    for col in ('nav', 'net_return', 'transaction_cost', 'cash', 'turnover'):
                        np.testing.assert_allclose(report[col], old[col], rtol=1e-11, atol=1e-12)
                if not all(audit[k] for k in ('liquidation_complete', 'turnover_budget_respected', 'cost_budget_respected')):
                    raise ValueError('Execution audit failed')
                trial = {'variant': name, 'cost_multiplier': int(m), **trial_metrics(report, trades, audit)}
                trials.append(trial)
                tag = f'{name}_cost{m}'
                report.to_csv(out/f'{tag}_returns.csv')
                trades.to_csv(out/f'{tag}_trades.csv', index=False)
                positions.to_csv(out/f'{tag}_positions.csv.gz', index=False)
                print(json.dumps({k: trial[k] for k in ('variant', 'cost_multiplier', 'cumulative_return', 'annualized_return',
                    'sharpe', 'max_drawdown', 'calendar_year_returns', 'passes_numeric_strategy_thresholds')}), flush=True)
        for name, expected in hashes.items():
            if file_fingerprint(Path(name)) != expected:
                raise ValueError('Input changed: '+name)
        verify_inputs(src, attribution)
        write('summary.json', {'scope': 'development_only_admission_bypassed', 'holdout_evaluated': False, 'trials': trials})
        write('status.json', {'status': 'completed_diagnostic'})
    except BaseException as exc:
        write('status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
