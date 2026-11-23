"""Replay single-factor template substitutions against the frozen weekly core."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admission import assess_weekly_substitution
from cn_portfolio import CNConfig, simulate_cn
from medium_frequency import load_member_bars
from alphax.factor_dsl import Candidate
from miner_lite import load_candidates
from run_cn_portfolio_research import load_quotes
from run_frozen_positive_holdout import audit_ledger
from run_medium_attribution import trial_metrics
from run_signal_stability import static_signals
from trader_lite import file_fingerprint

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"
SEEDS = {
    "mf_low_volatility_60": ("volatility", (60,)),
    "mf_low_downside_risk_60": ("downside", (60,)),
    "mf_momentum_120_skip20": ("momentum", (120, 20)),
    "mf_trend_60_120": ("trend", (60, 120)),
}
PARAMETERS = (
    ("mf_low_volatility_40", "mf_low_volatility_60", (40,)),
    ("mf_low_volatility_90", "mf_low_volatility_60", (90,)),
    ("mf_low_downside_risk_40", "mf_low_downside_risk_60", (40,)),
    ("mf_low_downside_risk_90", "mf_low_downside_risk_60", (90,)),
    ("mf_momentum_120_skip10", "mf_momentum_120_skip20", (120, 10)),
    ("mf_momentum_180_skip20", "mf_momentum_120_skip20", (180, 20)),
    ("mf_trend_40_120", "mf_trend_60_120", (40, 120)),
    ("mf_trend_60_180", "mf_trend_60_120", (60, 180)),
)


def template_candidates():
    variants = {}
    for name, control, parameters in PARAMETERS:
        family, original = SEEDS[control]
        edits = sum(before != after for before, after in zip(original, parameters, strict=True))
        if edits != 1:
            raise ValueError(f"{name} must change exactly one parameter")
        variants[name] = (
            Candidate(name, "feature", (name,), f"Test {family} with parameters {parameters}.", control_id=control),
            (family, parameters),
        )
    return variants


def make_features(raw, calendar, start, end, definitions):
    bars, intervals = load_member_bars(raw / "daily_pv.h5", raw / "qlib_csi300.txt", start, end)
    pieces = []
    for ticker, stock in bars.groupby("ticker", sort=True):
        dates = calendar[(calendar >= stock.date.min()) & (calendar <= end)]
        frame = stock.set_index("date").reindex(dates)
        columns = ["open", "close", "high", "low", "volume"]
        frame = frame[columns].where(frame[columns] > 0)
        close = frame.close
        returns = np.log(close).diff()
        values = {}
        for name, (family, parameters) in definitions.items():
            first = parameters[0]
            if family == "volatility":
                values[name] = -returns.rolling(first).std()
            elif family == "downside":
                values[name] = -np.sqrt(returns.clip(upper=0).pow(2).rolling(first).mean())
            elif family == "momentum":
                values[name] = close.shift(parameters[1]) / close.shift(first) - 1
            else:
                values[name] = close.rolling(first).mean() / close.rolling(parameters[1]).mean() - 1
        features = pd.DataFrame(values, index=dates).replace([np.inf, -np.inf], np.nan)
        features.loc[frame.isna().any(axis=1), :] = np.nan
        member = np.zeros(len(dates), dtype=bool)
        for period in intervals.loc[intervals.ticker == ticker].itertuples():
            member |= (dates >= period.start) & (dates <= period.end)
        features = features.loc[member & (dates >= start)].rename_axis("date").reset_index()
        features["ticker"] = ticker
        pieces.append(features)
    return pd.concat(pieces, ignore_index=True)


def evaluate(out, raw, calendar, window, definitions, portfolios, configs):
    start, end = window
    first = calendar[calendar >= start][0]
    previous = calendar[calendar.get_loc(first) - 1]
    frame = make_features(raw, calendar, previous, end, definitions)
    quotes = load_quotes(raw / "daily_pv.h5", set(frame.ticker), calendar, start, end)
    results = {}
    for name, groups in portfolios.items():
        signals = static_signals(frame, groups, calendar, first, start, end)
        results[name] = {}
        for multiplier, config in configs.items():
            report, trades, positions, audit = simulate_cn(signals, quotes, calendar, config, start, end)
            audit_ledger(report, trades, positions, calendar, config)
            report.to_csv(out / f"{name}_cost{multiplier}_returns.csv")
            results[name][multiplier] = trial_metrics(report, trades, audit)
        print(out.name, name, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True)
    raw = ML / "data/raw/qlib_csi300_quantaalpha"
    historical = ML / "data/qlib/alphacrafter_positive_holdout_20261117"
    frozen = json.loads((historical / "selection_frozen.json").read_text())
    core = frozen["selected"]["variant"]["groups"]
    candidates_path = HERE / "candidates_four_factor_core_v1.json"
    seeds = load_candidates(candidates_path, list(SEEDS))
    if {c.features[0] for c in seeds} != {feature for group in core for feature, _ in group}:
        raise ValueError("JSON seeds differ from frozen core")
    configs = {key: CNConfig(**value) for key, value in frozen["cost_scenarios"].items()}
    variants = template_candidates()
    definitions = {**SEEDS, **{name: spec for name, (_, spec) in variants.items()}}
    portfolios = {"core": core}
    for name, (candidate, _) in variants.items():
        portfolios[name] = [[(name if feature == candidate.control_id else feature, direction)
                             for feature, direction in group] for group in core]
    development = ("2022-01-01", "2023-12-31")
    replay = tuple(frozen["holdout"])

    def write(name, payload):
        (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    write("plan.json", {"development": development, "observed_replay": replay, "portfolios": portfolios,
                        "definitions": definitions, "controls": {n: c.control_id for n, (c, _) in variants.items()},
                        "cost_scenarios": frozen["cost_scenarios"],
                        "selection": "Highest worst development calendar-year net return across both costs; ties by identifier",
                        "inputs": [file_fingerprint(path) for path in
                                   [candidates_path, historical / "selection_frozen.json", Path(__file__),
                                    *(raw / name for name in ("daily_pv.h5", "qlib_csi300.txt", "qlib_day.txt"))]]})
    calendar = pd.DatetimeIndex(pd.to_datetime((raw / "qlib_day.txt").read_text().splitlines()))
    dev_out = out / "development"
    dev_out.mkdir()
    dev = evaluate(dev_out, raw, calendar, development, definitions, portfolios, configs)
    winner = min(variants, key=lambda name: (-min(ret for trial in dev[name].values()
                                                  for ret in trial["calendar_year_returns"].values()), name))
    write("selection.json", {"variant": winner, "control": variants[winner][0].control_id})
    replay_out = out / "replay"
    replay_out.mkdir()
    replay_results = evaluate(replay_out, raw, calendar, replay, definitions,
                              {"core": core, "substitution": portfolios[winner],
                               "template_only": [[(winner, 1)]]}, configs)
    for multiplier in configs:
        expected = pd.read_csv(historical / f"cost{multiplier}_returns.csv")
        actual = pd.read_csv(replay_out / f"core_cost{multiplier}_returns.csv")
        pd.testing.assert_series_equal(actual.date, expected.date)
        np.testing.assert_allclose(actual.nav, expected.nav, rtol=1e-11, atol=1e-12)
    write("summary.json", {"selected_variant": winner, "development": dev, "replay": replay_results,
                           "weekly_replacement_gate": assess_weekly_substitution(
                               replay_results["core"], replay_results["substitution"], independent=False),
                           "scope": "observed historical replay; no formal Miner admission"})


if __name__ == "__main__":
    main()
