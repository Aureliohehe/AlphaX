"""Recompute the frozen positive-return experiment from its saved artifacts."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from cn_portfolio import CNConfig
from run_frozen_positive_holdout import audit_ledger
from trader_lite import file_fingerprint, is_feasible, summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    read = lambda name: json.loads((root/name).read_text())
    require(read('status.json')['status'] == 'completed_research_holdout', 'Incomplete experiment')
    frozen, summary = read('selection_frozen.json'), read('summary.json')
    require(frozen['holdout'] == ['2024-01-01', '2025-12-31'], 'Unexpected reserved period')
    require(frozen['selected']['variant'] == summary['selected_variant'], 'Selected strategy changed')
    require(not summary['formal_strategy_admitted'], 'Research result incorrectly marked admitted')
    require(read('feature_audit.json')['prefix_matches_development'], 'Feature prefix mismatch')
    for name, expected in frozen['inputs'].items():
        path = Path(name)
        original = root/'source_snapshot'/path.name if path.name == 'run_frozen_positive_holdout.py' else path
        actual = file_fingerprint(original)
        require(actual['sha256'] == expected['sha256'], 'Frozen input changed: '+name)
    for path in root.glob('recovery_frozen*.json'):
        recovery = json.loads(path.read_text())
        require(recovery['original_selection'] == file_fingerprint(root/'selection_frozen.json'), 'Recovery changed selection')
        require(recovery['selection_rerun'] is False and recovery['holdout_returns_previously_computed'] is False,
                'Unexpected strategy/performance retry')
        require(any(file_fingerprint(p)['sha256'] == recovery['runner']['sha256']
                    for p in (root/'source_snapshot').glob('*.py')), 'Missing recovery source snapshot')
    for name, expected in read('artifact_audit.json')['artifacts'].items():
        # The first runner captured its running status before finalization.
        # Completed status and summary are checked and fingerprinted below.
        if name != 'status.json':
            require(file_fingerprint(root/name) == expected, 'Output artifact changed: '+name)
    calendar_path = Path(__file__).resolve().parent.parent/'machine-learn/data/raw/qlib_csi300_quantaalpha/qlib_day.txt'
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    expected_dates = calendar[(calendar >= '2024-01-01') & (calendar <= '2025-12-31')]
    derived = []
    require({t['cost_multiplier'] for t in summary['trials']} == {1, 2} and len(summary['trials']) == 2,
            'Missing or duplicated cost scenario')
    for trial in summary['trials']:
        m = str(trial['cost_multiplier'])
        report = pd.read_csv(root/f'cost{m}_returns.csv', parse_dates=['date']).set_index('date')
        trades = pd.read_csv(root/f'cost{m}_trades.csv', parse_dates=['date', 'signal_date'])
        positions = pd.read_csv(root/f'cost{m}_positions.csv.gz', parse_dates=['date'])
        pd.testing.assert_index_equal(report.index, expected_dates.rename('date'))
        config = CNConfig(**frozen['cost_scenarios'][m])
        audit = audit_ledger(report, trades, positions, calendar, config)
        metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
        for field in ('cumulative_return', 'annualized_return', 'sharpe', 'max_drawdown'):
            require(abs(metrics[field]-trial[field]) < 1e-11, 'Summary mismatch: '+field)
        years = {str(y): float((1+g.net_return).prod()-1) for y, g in report.groupby(report.index.year)}
        for year in ('2024', '2025'):
            require(abs(years[year]-trial['calendar_year_returns'][year]) < 1e-11, 'Year summary mismatch')
            require(years[year] > 0, 'Nonpositive annual return')
        require(metrics['cumulative_return'] > 0, 'Nonpositive net cumulative return')
        require(is_feasible(metrics) == trial['passes_numeric_strategy_thresholds'], 'Formal gate mismatch')
        derived.append({'cost_multiplier': int(m), **metrics, 'years': years, 'ledger_audit': audit})
    require(summary['positive_net_return_observed'] and summary['positive_at_both_costs'], 'Goal summary mismatch')
    verification = {'status': 'passed', 'auditor': file_fingerprint(Path(__file__)),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'scope': 'Full reserved-period research simulation; one frozen strategy and both frozen costs',
        'selection_and_recovery_preserved': True,
        'original_manifest_note': 'Original audit captured running status.json; all other original hashes match. This final manifest fingerprints completed status and summary.',
        'formal_strategy_admitted': False, 'derived_results': derived,
        'artifacts': {p.name: file_fingerprint(p) for p in root.iterdir() if p.is_file() and p.name != 'verification.json'}}
    (root/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps({'status': 'passed', 'net_returns': [r['cumulative_return'] for r in derived],
                      'sessions': len(expected_dates), 'selection_unchanged': True}))


if __name__ == '__main__':
    main()
