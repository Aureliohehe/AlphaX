"""Freeze research portfolios, validate net returns, then conditionally evaluate holdout."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from alphax.experiment import MINER_SOURCE_FILES
from alphax.factor_dsl import candidate_from_dict, evaluate_candidate_score
from cn_portfolio import CNConfig, simulate_cn
from trader_lite import file_fingerprint, is_feasible, select_configuration, summarize

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"


def write_report(out, summary):
    """Summarize actual cash fees; do not interpret sum of daily fee rates as initial-capital costs."""
    diagnostics = {}
    lines = ["# CSI300 persistent long-only portfolio research", "",
             "Validation: 2022–2023. Signals and three configurations frozen before evaluating returns. All figures are research-simulation results.", "",
             "| Stocks | Net cumulative | Net annualized | Sharpe (RF=0) | Max drawdown | Mean daily one-way turnover | Fees / initial capital |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for trial in summary["validation_results"]:
        name = trial["name"]
        trades = pd.read_csv(out / f"validation_{name}_trades.csv", parse_dates=["date", "signal_date"])
        returns = pd.read_csv(out / f"validation_{name}_returns.csv", parse_dates=["date"])
        cash_costs = {key: float(trades[key].sum()) for key in ("commission", "slippage", "stamp_duty")}
        costs = sum(cash_costs.values())
        years = {str(int(year)): float((1 + group.net_return).prod() - 1)
                 for year, group in returns.groupby(returns.date.dt.year)}
        inventory, bought = {}, {}
        for trade in trades.itertuples():
            if trade.signal_date >= trade.date:
                raise ValueError("Trade used a current/future close signal")
            if trade.side == "buy":
                inventory[trade.ticker] = inventory.get(trade.ticker, 0.) + trade.adjusted_units
                bought[trade.ticker] = trade.date
            else:
                if bought[trade.ticker] >= trade.date:
                    raise ValueError("Ledger violates T+1")
                inventory[trade.ticker] -= trade.adjusted_units
                if inventory[trade.ticker] < -1e-9:
                    raise ValueError("Ledger sold unowned adjusted units")
        diagnostics[name] = {"paid_cost_fraction_of_initial_capital": costs, "paid_cost_components": cash_costs,
                             "calendar_year_net_returns": years, "ledger_signal_and_T_plus_one_audit": "passed"}
        sharpe = "undefined" if trial["sharpe"] is None else f"{trial['sharpe']:.2f}"
        lines.append(f"| {trial['config']['topk']} | {trial['cumulative_return']:.2%} | {trial['annualized_return']:.2%} | {sharpe} | {trial['max_drawdown']:.2%} | {trial['mean_daily_turnover']:.1%} | {costs:.1%} |")
    benchmark = summary["validation_results"][0]["benchmark_cumulative_return"]
    lines += ["", f"CSI300 gross buy-and-hold benchmark over the same first-entry-open/last-exit-open interval: {benchmark:.2%}. Benchmark has 100% equity exposure; strategies target 80% equity/20% cash.", "",
              f"Selection: {summary['status']}; reserved-period evaluation: {summary['holdout_evaluated']}.",
              "If no strategy is feasible, the reserved period is not evaluated. This is not a claim of 0% traded holdout returns.", "",
              "Commission: 3bp and slippage: 5bp each side, both assumptions. Sell stamp duty: 10bp before 2023-08-28, 5bp thereafter. No interest paid on cash.",
              "Turnover = half of daily total traded notional / previous close NAV. Fees / initial capital comes from actual normalized cash ledger sums, not the sum of daily fee rates.", "",
              "Limitations:", *["- " + item for item in summary["limitations"]], "",
              "Frozen definitions and rule-source links: strategy_frozen.json. Net returns, trades and holdings: validation_top*_*.csv[.gz]. Cost/yearly/ledger checks: diagnostics.json."]
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return diagnostics


def composite_signals(features, pool, calendar, start, end):
    entries = pool["candidates_by_horizon"]["1"]
    if not entries:
        raise ValueError("No 1d research candidates")
    candidates = [(candidate_from_dict(r["definition"]), int(r["direction"])) for r in entries]
    fields = sorted({f for c, _ in candidates for f in c.features})
    pieces = []
    warmup = pd.Timestamp(start) - pd.Timedelta(days=730)
    for chunk in pd.read_csv(features, usecols=["date", "ticker", *fields], chunksize=100000, parse_dates=["date"]):
        pieces.append(chunk.loc[chunk.date.between(warmup, end)])
    frame = pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate feature keys")
    scores = []
    for candidate, direction in candidates:
        values = direction * evaluate_candidate_score(candidate, frame)
        scores.append(values.replace([np.inf, -np.inf], np.nan).groupby(frame.date, sort=False).rank(pct=True))
    score = pd.concat(scores, axis=1).sum(axis=1, min_count=len(scores)) / len(scores)
    frame["prediction"] = score
    # Exact previous market session, never previous available feature date.
    execution_sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    previous = dict(zip(calendar[1:], calendar[:-1]))
    wanted = {previous[d] for d in execution_sessions}
    return frame.loc[frame.date.isin(wanted), ["date", "ticker", "prediction"]].copy()


def load_quotes(source, tickers, calendar, start, end):
    import h5py
    first = calendar[calendar < pd.Timestamp(start)][-1]
    pieces = []
    with h5py.File(source) as hdf:
        g = hdf["data"]
        fields = [x.decode() for x in g["axis0"][:]]
        if fields[:2] != ["$open", "$close"]:
            raise ValueError("Unexpected price field order")
        dates = pd.DatetimeIndex(pd.to_datetime(g["axis1_level0"][:]))
        names = np.array([v.decode() for v in g["axis1_level1"][:]])
        dl, il = g["axis1_label0"][:], g["axis1_label1"][:]
        codes = np.isin(names, list(tickers | {"SH000300"}))
        for offset in range(0, len(dl), 250000):
            d, i = dl[offset:offset+250000], il[offset:offset+250000]
            mask = codes[i] & (dates[d] >= first) & (dates[d] <= pd.Timestamp(end))
            if mask.any():
                values = g["block0_values"][offset:offset+250000][mask]
                pieces.append(pd.DataFrame({"date": dates[d[mask]], "ticker": names[i[mask]],
                                            "open": values[:, 0], "close": values[:, 1]}))
    return pd.concat(pieces, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory")
    root = ML / "data/qlib/alphacrafter_csi300_research_pool_v1"
    source = ML / "data/raw/qlib_csi300_quantaalpha"
    feature_path = ML / "data/qlib/alphacrafter_csi300_frozen80_v1/features.csv.gz"
    pool = json.loads((root / "research_pool.json").read_text())
    plan = json.loads((root / "strategy_validation_plan.json").read_text())
    if pool["market"] != "cn" or pool["label_mode"] != "paper_close":
        raise ValueError("Expected the frozen CN research pool")
    if file_fingerprint(root / "research_pool.json") != plan["research_pool_fingerprint"]:
        raise ValueError("Research pool differs from validation plan")
    if pd.Timestamp(pool["research_end"]) >= pd.Timestamp(plan["validation"][0]):
        raise ValueError("Overlapping research and strategy validation")
    configs = [CNConfig(k) for k in (10, 30, 50)]
    calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
    out.mkdir(parents=True)

    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    sources = [root / "research_pool.json", root / "strategy_validation_plan.json", feature_path,
               source / "daily_pv.h5", source / "qlib_day.txt", Path(__file__), HERE / "cn_portfolio.py",
               *(HERE / name for name in MINER_SOURCE_FILES), HERE / "alphax/factor_dsl.py", HERE / "trader_lite.py"]
    fingerprints = {str(p): file_fingerprint(p) for p in sources}
    frozen = {"scope": "retrospective_adjusted_unit_A_share_portfolio_proxy_not_live_execution",
              "created_at": datetime.now(timezone.utc).isoformat(), "research_pool": fingerprints[str(root / "research_pool.json")],
              "horizon": 1, "factor_ids": [r["candidate_id"] for r in pool["candidates_by_horizon"]["1"]],
              "directions": [r["direction"] for r in pool["candidates_by_horizon"]["1"]],
              "weights": "equal weight across 13 directional cross-sectional percentile ranks; all components required",
              "configs": [asdict(c) for c in configs], "validation": plan["validation"], "holdout": plan["holdout"],
              "selection": "original portfolio gate; highest validation Sharpe among feasible; no feasible means no holdout evaluation",
              "execution": "T close signal; T+1 open rebalance; sell overnight inventory before buys; terminal open liquidation",
              "stamp_duty": {"before_2023_08_28_sell": .001, "from_2023_08_28_sell": .0005},
              "sources": fingerprints,
              "limitations": ["Fractional adjusted units, not actual 100-share lots; no minimum commission or depth/volume participation model",
                              "Opening +/-4.8% gap filters are conservative assumptions, not actual price-limit/ST reconstruction",
                              "No same-day volume, high/low, future returns or future tradability used for stock selection/fills",
                              "Missing open blocks trades; missing held close uses last known mark and is reported",
                              "Current pool contains correlated/monotonically equivalent factors; no adaptive weighting",
                              "Supplied historical membership and adjustment provenance are not independently PIT certified"],
              "rule_sources": ["https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html",
                               "https://www.chinatax.gov.cn/chinatax/n810219/n810780/c5211220/content.html",
                               "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml"]}
    write("strategy_frozen.json", frozen)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        if path.suffix == ".py":
            shutil.copyfile(path, snapshot / path.name)
    write("status.json", {"status": "evaluating_validation"})
    try:
        start, end = plan["validation"]
        signals = composite_signals(feature_path, pool, calendar, start, end)
        signals.to_csv(out / "validation_signals.csv.gz", index=False)
        quotes = load_quotes(source / "daily_pv.h5", set(signals.ticker), calendar, start, end)
        quotes.to_csv(out / "validation_quotes.csv.gz", index=False)
        trials = []
        for config in configs:
            name = f"top{config.topk}"
            report, trades, positions, audit = simulate_cn(signals, quotes, calendar, config, start, end)
            # No risk-free cash yield; Sharpe uses zero RF for this CN experiment.
            metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
            result = {"name": name, "config": asdict(config), **metrics, "execution_audit": audit}
            result["feasible"] = is_feasible(result) and audit["liquidation_complete"]
            trials.append(result)
            report.to_csv(out / f"validation_{name}_returns.csv")
            trades.to_csv(out / f"validation_{name}_trades.csv", index=False)
            positions.to_csv(out / f"validation_{name}_positions.csv.gz", index=False)
            print(json.dumps({"config": name, "cumulative_return": metrics["cumulative_return"],
                              "sharpe": metrics["sharpe"], "feasible": result["feasible"]}), flush=True)
        write("validation_results.json", trials)
        feasible = [r for r in trials if r["feasible"]]
        selected = select_configuration(feasible) if feasible else None
        write("selection_frozen.json", {"selected": selected, "status": "selected" if selected else "no_feasible_strategy"})
        selection_hash = file_fingerprint(out / "selection_frozen.json")
        holdout = None
        if selected:
            write("status.json", {"status": "evaluating_frozen_holdout"})
            start, end = plan["holdout"]
            hs = composite_signals(feature_path, pool, calendar, start, end)
            hq = load_quotes(source / "daily_pv.h5", set(hs.ticker), calendar, start, end)
            report, trades, positions, audit = simulate_cn(hs, hq, calendar, CNConfig(**selected["config"]), start, end)
            holdout = {**summarize(report, 1_000_000., annual_risk_free_rate=0.), "execution_audit": audit}
            report.to_csv(out / "holdout_returns.csv")
            trades.to_csv(out / "holdout_trades.csv", index=False)
            positions.to_csv(out / "holdout_positions.csv.gz", index=False)
        if file_fingerprint(out / "selection_frozen.json") != selection_hash:
            raise ValueError("Selection changed during holdout")
        for path, expected in fingerprints.items():
            if file_fingerprint(Path(path)) != expected:
                raise ValueError("Input/code changed: " + path)
        summary = {"scope": frozen["scope"], "validation_results": trials, "selected_config": selected["name"] if selected else None,
                   "status": "selected" if selected else "no_feasible_strategy", "holdout": holdout,
                   "holdout_evaluated": selected is not None, "limitations": frozen["limitations"]}
        write("summary.json", summary)
        write_report(out, summary)
        write("status.json", {"status": "completed_research"})
        print(json.dumps({"status": summary["status"], "holdout_evaluated": summary["holdout_evaluated"]}), flush=True)
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
