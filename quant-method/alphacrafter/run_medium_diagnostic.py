"""Exploratory 5/20-session returns with all frozen factors, bypassing admission."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pandas as pd

from cn_portfolio import simulate_cn
from medium_frequency import definitions, screen_signals, strategy_presets
from run_cn_portfolio_research import load_quotes
from trader_lite import file_fingerprint, is_feasible, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    src, out = args.research_dir.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory")
    read = lambda name: json.loads((src / name).read_text())
    manifest = read("manifest.json")
    if read("status.json")["status"] != "completed_research" or manifest["visible_end"] != "2023-12-31":
        raise ValueError("Need completed medium-frequency research through 2023 only")
    if read("definitions_frozen.json") != definitions():
        raise ValueError("Frozen definitions differ from current code")
    for name, expected in manifest["sources"].items():
        if file_fingerprint(Path(name)) != expected:
            raise ValueError("Research input or code changed: " + name)
    pool = read("research_pool.json")
    # Retain every predefined factor. No ranking/selection by development or
    # strategy returns; train-only directions remain separate for each horizon.
    assessments = read("research_assessments.json")
    lookup = {d["candidate_id"]: d for d in definitions()}
    for h in (5, 20):
        entries = []
        for row in assessments:
            if row["horizon"] != h:
                continue
            if row["direction"] not in (-1, 1):
                raise ValueError("Undefined training direction")
            entries.append({"candidate_id": row["candidate_id"], "direction": row["direction"],
                            "definition": lookup[row["candidate_id"]], "strategy_admitted": False,
                            "status": "diagnostic_only_admission_bypassed"})
        if len(entries) != len(lookup):
            raise ValueError("Incomplete candidate set")
        pool["candidates_by_horizon"][str(h)] = entries
    pool["scope"] = "diagnostic_only_admission_bypassed"
    pool["eligible_counts"] = read("research_pool.json")["eligible_counts"]
    out.mkdir(parents=True)
    write = lambda name, value: (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write("status.json", {"status": "running"})
    source = Path(__file__).resolve().parent.parent / "machine-learn/data/raw/qlib_csi300_quantaalpha"
    calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
    presets = strategy_presets()
    hashes = {str(p): file_fingerprint(p) for p in [src / "features_labels.csv.gz", src / "research_assessments.json",
                                                  src / "manifest.json", Path(__file__)]}
    frozen = {"scope": pool["scope"], "admission_bypassed": True, "pool": pool,
              "configs": {str(h): asdict(c) for h, c in presets.items()}, "inputs": hashes,
              "screening": manifest["screening"], "window": ["2022-01-01", "2023-12-31"],
              "strategy_window_previously_examined": True, "holdout_evaluated": False}
    write("diagnostic_frozen.json", frozen)
    shutil.copyfile(__file__, out / Path(__file__).name)
    try:
        frame = pd.read_csv(src / "features_labels.csv.gz")
        for col in frame:
            if col == "date" or col.startswith(("label_start_", "label_end_")):
                frame[col] = pd.to_datetime(frame[col])
        if frame.date.max() > pd.Timestamp("2023-12-31"):
            raise ValueError("Feature frame extends into reserved period")
        trials = []
        for h, config in presets.items():
            print(f"Screening {h}d: all {len(lookup)} frozen factors", flush=True)
            signals, selections = screen_signals(frame, pool, h, calendar, calendar[calendar >= "2022-01-01"][0],
                                                  "2022-01-01", "2023-12-31")
            signals.to_csv(out / f"signals_{h}d.csv.gz", index=False)
            selections.to_csv(out / f"screening_{h}d.csv", index=False)
            if signals.empty:
                raise ValueError("No diagnostic signals")
            quotes = load_quotes(source / "daily_pv.h5", set(signals.ticker), calendar, "2022-01-01", "2023-12-31")
            report, trades, positions, audit = simulate_cn(signals, quotes, calendar, config, "2022-01-01", "2023-12-31")
            metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
            years = {str(y): float((1 + g.net_return).prod() - 1) for y, g in report.groupby(report.index.year)}
            trial = {"horizon": h, **metrics, "calendar_year_returns": years,
                     "paid_cost_fraction_of_initial_capital": float(trades.total_cost.sum()),
                     "execution_audit": audit, "passes_numeric_strategy_thresholds": is_feasible(metrics),
                     "strategy_admitted": False, "config": asdict(config)}
            trials.append(trial)
            report.to_csv(out / f"returns_{h}d.csv")
            trades.to_csv(out / f"trades_{h}d.csv", index=False)
            positions.to_csv(out / f"positions_{h}d.csv.gz", index=False)
            print(json.dumps(trial, ensure_ascii=False), flush=True)
        for name, expected in hashes.items():
            if file_fingerprint(Path(name)) != expected:
                raise ValueError("Diagnostic input changed: " + name)
        write("summary.json", {"scope": pool["scope"], "admission_bypassed": True,
                               "holdout_evaluated": False, "strategies": trials})
        lines = ["# 中低频探索性收益诊断", "", "全部 12 个冻结因子绕过准入；方向只取训练期，保留成熟标签动态筛选。",
                 "2022–2023 为已看过的迭代研究窗口，未评估 2024–2025；不构成正式策略准入。", "",
                 "| 调仓间隔 | 扣费累计收益 | 年化 | Sharpe（RF=0） | 最大回撤 | 日均单边换手 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for t in trials:
            sharpe = "—" if t["sharpe"] is None else f"{t['sharpe']:.2f}"
            lines.append(f"| {t['horizon']} 日 | {t['cumulative_return']:.2%} | {t['annualized_return']:.2%} | {sharpe} | {t['max_drawdown']:.2%} | {t['mean_daily_turnover']:.2%} |")
        lines += ["", "目标 30 只、80% 股票仓位；每侧佣金 3bp、滑点 5bp，卖出印花税 10/5bp；现金无利息。",
                  "基准为 100% 沪深300指数毛收益，策略为80%目标股票敞口，敞口与费用口径不同。",
                  "所有原研究数据和模拟成交限制仍适用；执行审计、年度收益与费用见 summary.json。"]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        write("status.json", {"status": "completed_diagnostic"})
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
