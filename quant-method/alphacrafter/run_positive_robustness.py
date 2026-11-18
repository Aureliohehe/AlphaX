"""Replay the positive development ablation under frozen execution cost stress."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cn_portfolio import CNConfig, simulate_cn
from run_cn_portfolio_research import load_quotes
from run_medium_attribution import trial_metrics
from trader_lite import file_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    src, out = args.source_dir.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error('Use a new output directory')
    if json.loads((src/'status.json').read_text())['status'] != 'completed_diagnostic':
        raise ValueError('Require completed attribution')
    summary = json.loads((src/'summary.json').read_text())
    trial = next(t for t in summary['trials'] if t['horizon'] == 5 and t['variant'] == 'fill_without_reversal')
    config = CNConfig(**trial['config'])
    raw = Path(__file__).resolve().parent.parent/'machine-learn/data/raw/qlib_csi300_quantaalpha'
    paths = [src/n for n in ('summary.json', '5d_fill_without_reversal_signals.csv.gz',
                             '5d_fill_without_reversal_returns.csv')]
    paths += [raw/'daily_pv.h5', raw/'qlib_day.txt']
    paths += [Path(__file__).resolve().parent/n for n in ('run_positive_robustness.py', 'cn_portfolio.py',
              'run_cn_portfolio_research.py', 'run_medium_attribution.py', 'trader_lite.py')]
    hashes = {str(p): file_fingerprint(p) for p in paths}
    configs = {str(m): replace(config, commission=config.commission*m, slippage=config.slippage*m)
               for m in (1, 2, 3)}
    out.mkdir(parents=True)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('frozen.json', {'scope': 'development_only_post_selection_stress', 'holdout_evaluated': False,
          'window': ['2022-01-01', '2023-12-31'], 'inputs': hashes,
          'configs': {m: asdict(c) for m, c in configs.items()},
          'selection_bias': 'Source variant chosen after observing development returns; no independent validation claim'})
    write('status.json', {'status': 'running'})
    try:
        signals = pd.read_csv(paths[1], parse_dates=['date'])
        calendar = pd.DatetimeIndex(pd.to_datetime((raw/'qlib_day.txt').read_text().splitlines()))
        quotes = load_quotes(raw/'daily_pv.h5', set(signals.ticker), calendar, '2022-01-01', '2023-12-31')
        results = []
        for multiplier, cfg in configs.items():
            report, trades, positions, audit = simulate_cn(signals, quotes, calendar, cfg, '2022-01-01', '2023-12-31')
            if multiplier == '1':
                original = pd.read_csv(paths[2], parse_dates=['date']).set_index('date')
                for column in ('nav', 'net_return', 'transaction_cost', 'cash', 'turnover'):
                    np.testing.assert_allclose(report[column], original[column], rtol=1e-11, atol=1e-12)
            if not all(audit[k] for k in ('liquidation_complete', 'turnover_budget_respected', 'cost_budget_respected')):
                raise ValueError('Execution audit failed')
            result = {'cost_multiplier': int(multiplier), **trial_metrics(report, trades, audit)}
            results.append(result)
            report.to_csv(out/f'returns_cost{multiplier}.csv')
            trades.to_csv(out/f'trades_cost{multiplier}.csv', index=False)
            print(json.dumps(result), flush=True)
        for name, expected in hashes.items():
            if file_fingerprint(Path(name)) != expected:
                raise ValueError('Input changed: '+name)
        write('summary.json', {'holdout_evaluated': False, 'trials': results})
        write('status.json', {'status': 'completed_diagnostic'})
    except BaseException as exc:
        write('status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
