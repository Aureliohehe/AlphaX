"""Frozen development-only execution and factor-group ablations; no admission."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cn_portfolio import simulate_cn
from medium_frequency import definitions, screen_signals, strategy_presets
from run_cn_portfolio_research import load_quotes
from trader_lite import file_fingerprint, is_feasible, summarize

HERE = Path(__file__).resolve().parent
DEFENSIVE = ["mf_low_volatility_60", "mf_low_downside_risk_60"]
REVERSAL = ["mf_momentum_60_skip5", "mf_trend_20_60", "mf_risk_adjusted_momentum_60",
            "mf_drawdown_60", "mf_volume_confirmed_trend", "mf_trend_consistency_60", "mf_reversal_20"]
TREND = ["mf_momentum_120_skip20", "mf_trend_60_120"]
GROUPS = {"defensive": DEFENSIVE, "reversal": REVERSAL, "trend": TREND}


def verify_research(src):
    """Check original source snapshots, immutable data, and unchanged factor code.

    cn_portfolio.py is deliberately changed for this experiment. Its original
    snapshot must match the research manifest; its new code is frozen separately.
    """
    read = lambda name: json.loads((src / name).read_text())
    manifest = read("manifest.json")
    if read("status.json")["status"] != "completed_research" or manifest["visible_end"] != "2023-12-31":
        raise ValueError("Require completed development-only medium research")
    if read("definitions_frozen.json") != definitions():
        raise ValueError("Factor definitions changed")
    for name, expected in manifest["sources"].items():
        path = Path(name)
        if path.suffix == ".py":
            snapshot_hash = file_fingerprint(src / "source_snapshot" / path.name)
            if any(snapshot_hash[k] != expected[k] for k in ("sha256", "bytes")):
                raise ValueError("Research source snapshot changed: " + name)
            if path.name != "cn_portfolio.py" and file_fingerprint(path) != expected:
                raise ValueError("Research factor/screening code changed: " + name)
        elif file_fingerprint(path) != expected:
            raise ValueError("Research raw input changed: " + name)
    return read, manifest


def trial_metrics(report, trades, audit):
    metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
    matched = report.copy()
    matched["net_return"] = report.matched_index_net_return
    bm = summarize(matched, 1_000_000., annual_risk_free_rate=0.)
    return {**metrics, "mean_equity_exposure": float((1-report.cash/report.nav).mean()),
            "mean_holdings": float(report.held_assets.mean()),
            "matched_index_net_cumulative": bm["cumulative_return"],
            "matched_index_net_annualized": bm["annualized_return"],
            "matched_index_net_max_drawdown": bm["max_drawdown"],
            "matched_excess_cumulative_pp": 100*(metrics["cumulative_return"]-bm["cumulative_return"]),
            "paid_cost_fraction_of_initial_capital": float(trades.total_cost.sum()),
            "calendar_year_returns": {str(y): float((1+g.net_return).prod()-1)
                                      for y, g in report.groupby(report.index.year)},
            "execution_audit": audit, "passes_numeric_strategy_thresholds": is_feasible(metrics),
            "strategy_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    src, baseline, out = args.research_dir.resolve(), args.baseline_dir.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory")
    read, manifest = verify_research(src)
    if json.loads((baseline / "status.json").read_text())["status"] != "completed_diagnostic":
        raise ValueError("Need completed original diagnostic for paired replay")
    pool = read("research_pool.json")
    lookup = {d["candidate_id"]: d for d in definitions()}
    rows = read("research_assessments.json")
    for h in (5, 20):
        entries = [{"candidate_id": r["candidate_id"], "direction": r["direction"],
                    "definition": lookup[r["candidate_id"]], "status": "diagnostic_only_admission_bypassed"}
                   for r in rows if r["horizon"] == h]
        if len(entries) != 12 or any(e["direction"] not in (-1, 1) for e in entries):
            raise ValueError("Incomplete train directions")
        directions = {e["candidate_id"]: e["direction"] for e in entries}
        if any(directions[f] != 1 for f in DEFENSIVE+TREND+["mf_reversal_20"]) or any(
                directions[f] != -1 for f in REVERSAL if f != "mf_reversal_20"):
            raise ValueError("Frozen economic groups differ from train directions")
        pool["candidates_by_horizon"][str(h)] = entries
    presets = strategy_presets()
    variants = [("original_all", False, []), ("fill_all", True, [])] + [
        ("fill_without_"+g, True, factors) for g, factors in GROUPS.items()]
    paths = [src / n for n in ("features_labels.csv.gz", "research_assessments.json", "manifest.json")]
    paths += [HERE / n for n in ("run_medium_attribution.py", "cn_portfolio.py", "medium_frequency.py",
                                "trader_lite.py", "run_cn_portfolio_research.py")]
    paths += [baseline / f"returns_{h}d.csv" for h in (5, 20)]
    hashes = {str(p): file_fingerprint(p) for p in paths}
    out.mkdir(parents=True)
    write = lambda n, v: (out / n).write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    frozen = {"scope": "development_only_admission_bypassed", "holdout_evaluated": False,
              "window": ["2022-01-01", "2023-12-31"], "groups": GROUPS, "pool": pool,
              "variants": [{"name": n, "fill_target_daily": mode, "excluded": exclude} for n, mode, exclude in variants],
              "presets": {str(h): asdict(c) for h, c in presets.items()}, "input_hashes": hashes,
              "fill_rule": "Scheduled target share quantities; outstanding buys on intervening opens; top up retained holdings at scheduled decisions only; no off-schedule sells; existing budget/holding/exit guards retained",
              "benchmark_rule": "Index overnight return times previous close risky weight plus index intraday return times post-trade risky value / previous NAV; subtract identical realized daily strategy cost / previous NAV",
              "benchmark_limitation": "Path-conditioned attribution proxy; uses realized strategy weights and fee rates, not independently executable index strategy or risk-adjusted alpha",
              "screening": manifest["screening"], "selection": "No return-based winner selection or holdout evaluation"}
    write("experiment_frozen.json", frozen)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for p in paths:
        if p.suffix == ".py":
            shutil.copyfile(p, snapshot / p.name)
    write("status.json", {"status": "running"})
    try:
        frame = pd.read_csv(src / "features_labels.csv.gz")
        for col in frame:
            if col == "date" or col.startswith(("label_start_", "label_end_")):
                frame[col] = pd.to_datetime(frame[col])
        if frame.date.max() > pd.Timestamp("2023-12-31"):
            raise ValueError("Reserved-period features")
        source = HERE.parent / "machine-learn/data/raw/qlib_csi300_quantaalpha"
        calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
        quotes = load_quotes(source / "daily_pv.h5", set(frame.ticker), calendar, "2022-01-01", "2023-12-31")
        trials = []
        for h, preset in presets.items():
            all_signals = None
            for name, mode, excluded in variants:
                print(f"Running {h}d {name}", flush=True)
                chosen_pool = deepcopy(pool)
                chosen_pool["candidates_by_horizon"][str(h)] = [e for e in pool["candidates_by_horizon"][str(h)]
                                                                if e["candidate_id"] not in excluded]
                if not excluded and all_signals is not None:
                    signals = all_signals
                else:
                    signals, selections = screen_signals(frame, chosen_pool, h, calendar,
                        calendar[calendar >= "2022-01-01"][0], "2022-01-01", "2023-12-31")
                    if signals.empty:
                        raise ValueError("Empty signals: " + name)
                    selections.to_csv(out / f"{h}d_{name}_screening.csv", index=False)
                    signals.to_csv(out / f"{h}d_{name}_signals.csv.gz", index=False)
                    if not excluded:
                        all_signals = signals
                config = replace(preset, fill_target_daily=mode)
                report, trades, positions, audit = simulate_cn(signals, quotes, calendar, config, "2022-01-01", "2023-12-31")
                if name == "original_all":
                    old = pd.read_csv(baseline / f"returns_{h}d.csv", parse_dates=["date"]).set_index("date")
                    for col in ("nav", "net_return", "turnover", "transaction_cost", "cash"):
                        np.testing.assert_allclose(report[col], old[col], rtol=1e-11, atol=1e-12)
                if not all(audit[k] for k in ("liquidation_complete", "turnover_budget_respected", "cost_budget_respected")):
                    raise ValueError("Execution audit failed: " + name)
                trial = {"horizon": h, "variant": name, "config": asdict(config), **trial_metrics(report, trades, audit)}
                trials.append(trial)
                report.to_csv(out / f"{h}d_{name}_returns.csv")
                trades.to_csv(out / f"{h}d_{name}_trades.csv", index=False)
                positions.to_csv(out / f"{h}d_{name}_positions.csv.gz", index=False)
                print(json.dumps({k: trial[k] for k in ("horizon", "variant", "cumulative_return", "mean_equity_exposure", "matched_excess_cumulative_pp")}), flush=True)
        for name, expected in hashes.items():
            if file_fingerprint(Path(name)) != expected:
                raise ValueError("Input changed: " + name)
        verify_research(src)
        write("summary.json", {"scope": frozen["scope"], "holdout_evaluated": False, "trials": trials})
        lines = ["# 中低频仓位与分组剔除诊断", "", "2022–2023 开发窗口；绕过准入；全部配置事先冻结；未评估留出。", "",
                 "| 周期 | 配置 | 净累计 | 年化 | 最大回撤 | 平均仓位 | 匹配基准净累计 | 差值/百分点 | 实付费用/初始本金 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for t in trials:
            lines.append(f"| {t['horizon']}d | {t['variant']} | {t['cumulative_return']:.2%} | {t['annualized_return']:.2%} | {t['max_drawdown']:.2%} | {t['mean_equity_exposure']:.2%} | {t['matched_index_net_cumulative']:.2%} | {t['matched_excess_cumulative_pp']:.2f} | {t['paid_cost_fraction_of_initial_capital']:.2%} |")
        lines += ["", frozen["fill_rule"], "", frozen["benchmark_rule"], "", frozen["benchmark_limitation"],
                  "", "剔除后重新等权并重新选股/筛选，效应包含权重、持仓与成本变化，不是线性因子收益归因。",
                  "原模拟数据来源、复权可分割单位、近似交易限制等局限继续适用。"]
        (out / "report.md").write_text("\n".join(lines)+"\n")
        write("status.json", {"status": "completed_diagnostic"})
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
