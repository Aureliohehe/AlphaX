"""Combine a frozen four-factor core and approved CSI300 Agent signals in separate sleeves."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cn_portfolio import CNConfig, simulate_cn
from run_cn_portfolio_research import load_quotes
from run_frozen_positive_holdout import ML, audit_ledger
from trader_lite import file_fingerprint, summarize


def combine_sleeves(core, agent, agent_weight):
    if not np.isfinite(agent_weight) or not 0 <= agent_weight <= 1:
        raise ValueError("Agent weight must be between 0 and 1")
    if not core.index.equals(agent.index) or not np.allclose(core.benchmark_return, agent.benchmark_return):
        raise ValueError("Sleeves must use the same sessions and CSI300 benchmark")
    core_capital = (1 - agent_weight) * core.nav
    agent_capital = agent_weight * agent.nav
    nav = core_capital + agent_capital
    previous_nav = nav.shift(1, fill_value=1.)
    core_weight = core_capital.shift(1, fill_value=1 - agent_weight) / previous_nav
    agent_weight_daily = agent_capital.shift(1, fill_value=agent_weight) / previous_nav
    fees = core_weight * core.transaction_cost + agent_weight_daily * agent.transaction_cost
    net = nav / previous_nav - 1
    return pd.DataFrame({
        "nav": nav, "net_return": net, "gross_return": net + fees,
        "transaction_cost": fees, "benchmark_return": core.benchmark_return,
        "turnover": core_weight * core.turnover + agent_weight_daily * agent.turnover,
        "core_capital": core_capital, "agent_capital": agent_capital,
        "agent_opening_weight": agent_weight_daily,
        "core_net_return": core.net_return, "agent_net_return": agent.net_return,
    })


def read_agent_signals(path, members, signal_sessions):
    if path is None:
        return pd.DataFrame(columns=["date", "ticker", "prediction"])
    signals = pd.read_csv(path).rename(columns={"datetime": "date", "instrument": "ticker"})
    signals = signals[["date", "ticker", "prediction"]].copy()
    signals["date"] = pd.to_datetime(signals.date)
    signals["prediction"] = pd.to_numeric(signals.prediction)
    if signals.duplicated(["date", "ticker"]).any() or not np.isfinite(signals.prediction).all():
        raise ValueError("Agent signals must have unique keys and finite scores")
    if not signals.date.isin(signal_sessions).all():
        raise ValueError("Agent scores must be dated on prior-close signal sessions for this window")
    if not pd.MultiIndex.from_frame(signals[["date", "ticker"]]).isin(members).all():
        raise ValueError("Agent scores include securities outside the historical CSI300 universe")
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--approved-agent-signals", type=Path)
    inputs.add_argument("--agent-cash", action="store_true")
    parser.add_argument("--agent-weight", type=float, default=.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.agent_weight) or not 0 <= args.agent_weight <= 1:
        parser.error("--agent-weight must be between 0 and 1")

    root = args.core_dir
    raw = ML / "data/raw/qlib_csi300_quantaalpha"
    frozen_path = root / "selection_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if (json.loads((root / "status.json").read_text())["status"] != "completed_research_holdout"
            or frozen["selected"]["variant"]["name"] != "static_balanced4"):
        raise ValueError("Require a completed static_balanced4 core experiment")
    start, end = frozen["holdout"]
    calendar_path = raw / "qlib_day.txt"
    quote_path = raw / "daily_pv.h5"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    signal_sessions = calendar[calendar.get_indexer(sessions) - 1]
    feature_path = root / "features.csv.gz"
    members = pd.MultiIndex.from_frame(
        pd.read_csv(feature_path, usecols=["date", "ticker"], parse_dates=["date"])[["date", "ticker"]]
    )
    signals = read_agent_signals(args.approved_agent_signals, members, signal_sessions)
    quotes = load_quotes(quote_path, set(signals.ticker), calendar, start, end)

    out = args.output_dir
    out.mkdir(parents=True)
    costs = frozen["cost_scenarios"]
    core_returns = {key: root / f"cost{key}_returns.csv" for key in costs}
    sources = [frozen_path, feature_path, calendar_path, quote_path, *core_returns.values()]
    if args.approved_agent_signals:
        sources.append(args.approved_agent_signals)
    manifest = {
        "scope": "historical_research_blend",
        "initial_agent_weight": args.agent_weight,
        "initial_core_weight": 1 - args.agent_weight,
        "allocation": "Initial split with no subsequent transfers; sleeve weights drift with NAV",
        "agent_input": "cash" if signals.empty else "upstream-approved close scores",
        "window": [start, end],
        "execution_settings": costs,
        "inputs": [file_fingerprint(path) for path in sources],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trials = []
    for key, settings in costs.items():
        core = pd.read_csv(core_returns[key], parse_dates=["date"], index_col="date")
        config = CNConfig(**settings)
        agent, trades, positions, _ = simulate_cn(signals, quotes, calendar, config, start, end)
        # Empty simulator ledgers have object columns. Give the auditor numeric arrays.
        trades = trades.astype({field: float for field in
                                ("commission", "slippage", "stamp_duty", "notional_nav_units", "total_cost")})
        positions = positions.astype({"adjusted_units": float, "mark": float})
        trades["date"] = pd.to_datetime(trades.date)
        positions["date"] = pd.to_datetime(positions.date)
        audit = audit_ledger(agent, trades, positions, calendar, config)
        combined = combine_sleeves(core, agent, args.agent_weight)
        combined.to_csv(out / f"cost{key}_combined_returns.csv")
        agent.to_csv(out / f"cost{key}_agent_returns.csv")
        trades.to_csv(out / f"cost{key}_agent_trades.csv", index=False)
        positions.to_csv(out / f"cost{key}_agent_positions.csv", index=False)
        trials.append({
            "cost_multiplier": int(key),
            "agent_ledger_audit": audit,
            "agent_return_contribution": float(args.agent_weight * (agent.nav.iloc[-1] - 1)),
            **{name: summarize(report, 1., 0.) for name, report in
               (("core", core), ("agent", agent), ("combined", combined))},
        })
    result = json.dumps({"trials": trials}, indent=2, allow_nan=False) + "\n"
    (out / "summary.json").write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
