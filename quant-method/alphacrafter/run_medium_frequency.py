"""Separate 5/20-session Miner/Screener/Trader research; never reads holdout performance."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from admission import build_research_pool, strategy_plan
from cn_portfolio import simulate_cn
from medium_frequency import (HORIZONS, FACTOR_HYPOTHESES, anchor_mask, definitions, factor_panel,
                              load_member_bars, screen_signals, strategy_presets)
from miner_lite import evaluate_score, metric_dict
from run_cn_portfolio_research import load_quotes
from trader_lite import file_fingerprint, is_feasible, summarize

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-strategies", action="store_true", help="Run fixed weekly/monthly configurations on already-examined 2022–2023; no holdout evaluation")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory")
    source = ML / "data/raw/qlib_csi300_quantaalpha"
    # Only the research/development window is loaded unless strategy evaluation
    # is requested. 2024+ raw prices/features/labels are never loaded here.
    visible_end = "2023-12-31" if args.evaluate_strategies else "2021-12-31"
    calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
    first_execution = calendar[calendar >= "2022-01-01"][0]
    presets = strategy_presets()
    out.mkdir(parents=True)

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def write(name, data):
        (out / name).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    write("status.json", {"status": "preparing"})
    inputs = [source / "daily_pv.h5", source / "qlib_csi300.txt", source / "qlib_day.txt",
              Path(__file__), HERE / "medium_frequency.py", HERE / "cn_portfolio.py", HERE / "admission.py",
              HERE / "miner_lite.py", HERE / "trader_lite.py", HERE / "run_cn_portfolio_research.py"]
    hashes = {str(p): file_fingerprint(p) for p in inputs}
    manifest = {"scope": "retrospective_medium_frequency_research_not_live_execution",
                "created_at": datetime.now(timezone.utc).isoformat(), "sources": hashes,
                "definitions": definitions(), "horizons": list(HORIZONS), "visible_end": visible_end,
                "label_mode": "execution_open_open", "label_rule": "T-close signal; entry T+1 open; exit T+h+1 open",
                "research_train": ["2010-01-01", "2018-12-31"], "research_development": ["2019-01-01", "2021-12-31"],
                "strategy_iteration": ["2022-01-01", "2023-12-31"], "final_reserved": ["2024-01-01", "2025-12-31"],
                "strategy_iteration_previously_examined": True,
                "research_sampling": "fixed nonoverlapping horizon anchors aligned to the frozen execution calendar phase",
                "screening": {"lookback_completed_periods": 12, "minimum_completed_periods": 6, "min_mean_rank_ic": 0.},
                "presets": {str(h): asdict(c) for h, c in presets.items()}, "holdout_evaluated": False,
                "limitations": ["5/20 sessions approximate weekly/monthly intervals, not civil-calendar weeks/months",
                                "Supplied member/calendar/adjustment provenance not independently PIT certified",
                                "Adjusted fractional units and approximate opening-gap filters; no actual lots, depth or ST limit reconstruction",
                                "Exposure is a target; holding locks/blocked exits and market drift can exceed it, while new buys have an exposure budget",
                                "Phase and terminal wind-down are frozen; factor rank turnover is not portfolio turnover",
                                "Research admission remains exploratory; 2022–2023 strategy results are iteration data, not fresh independent validation"]}
    write("manifest.json", manifest)
    write("definitions_frozen.json", definitions())
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for p in inputs:
        if p.suffix == ".py":
            shutil.copyfile(p, snapshots / p.name)
    try:
        bars, intervals = load_member_bars(source / "daily_pv.h5", source / "qlib_csi300.txt", "2010-01-01", visible_end)
        parts, expected = [], 0
        groups = bars.groupby("ticker", sort=True)
        for i, (ticker, raw) in enumerate(groups, 1):
            values = factor_panel(raw, calendar, end=visible_end)
            membership = intervals[intervals.ticker == ticker]
            keep = pd.Series(False, index=values.index)
            for row in membership.itertuples():
                keep |= values.date.between(row.start, row.end)
            keep &= values.date.between("2010-01-01", visible_end)
            parts.append(values.loc[keep])
            if i % 100 == 0:
                print(f"Prepared {i}/{len(groups)} medium-frequency stock histories", flush=True)
        frame = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        del bars, parts
        sessions = calendar[(calendar >= "2010-01-01") & (calendar <= visible_end)]
        for _, group in intervals.groupby("ticker"):
            mask = np.zeros(len(sessions), dtype=bool)
            for row in group.itertuples():
                mask |= (sessions >= row.start) & (sessions <= row.end)
            expected += int(mask.sum())
        if frame.duplicated(["date", "ticker"]).any() or len(frame) != expected:
            raise ValueError("Medium-frequency panel does not match supplied historical member rows")
        frame.to_csv(out / "features_labels.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
        write("feature_audit.json", {"rows": len(frame), "expected_membership_rows": expected,
                                     "max_visible_signal_date": str(frame.date.max().date()), "features": list(FACTOR_HYPOTHESES)})
        write("status.json", {"status": "researching_factors"})
        results = []
        for definition in definitions():
            name = definition["candidate_id"]
            metrics = {}
            for h in HORIZONS:
                anchors = anchor_mask(frame.date, calendar, first_execution, h)
                metrics[str(h)] = {}
                for split, start, end in (("train", "2010-01-01", "2018-12-31"), ("validation", "2019-01-01", "2021-12-31")):
                    label_end = frame[f"label_end_{h}d"]
                    mask = anchors & frame.date.between(start, end) & label_end.notna() & label_end.le(pd.Timestamp(end))
                    m = metric_dict(evaluate_score(frame.date, frame[name], frame[f"target_open_open_{h}d"], mask, frame.ticker))
                    m["sampled_signal_days"] = int(frame.loc[mask, "date"].nunique())
                    metrics[str(h)][split] = m
            results.append({"candidate_id": name, "horizons": metrics})
            print(f"Evaluated {name}", flush=True)
        audit = {"as_of": "2021-12-31", "label_mode": "execution_open_open",
                 "windows": {"train": ["2010-01-01", "2018-12-31"], "validation": ["2019-01-01", "2021-12-31"]}}
        write("research_temporal_audit.json", audit)
        pool, assessments = build_research_pool(results, definitions(), audit, "cn", horizons=HORIZONS)
        write("factor_metrics.json", results)
        write("research_pool.json", pool)
        write("research_assessments.json", assessments)
        write("strategy_validation_plan.json", strategy_plan(pool, "2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"))
        write("strategy_frozen.json", {"research_pool": file_fingerprint(out / "research_pool.json"),
                                        "presets": manifest["presets"], "screening": manifest["screening"],
                                        "iteration_window": manifest["strategy_iteration"], "reserved_window": manifest["final_reserved"]})
        trials = []
        if args.evaluate_strategies:
            write("status.json", {"status": "evaluating_frozen_strategy_iterations"})
            for h, config in presets.items():
                if not pool["candidates_by_horizon"][str(h)]:
                    trials.append({"horizon": h, "status": "not_run_no_research_candidates"})
                    continue
                signals, selections = screen_signals(frame, pool, h, calendar, first_execution, "2022-01-01", "2023-12-31")
                signals.to_csv(out / f"signals_{h}d.csv.gz", index=False)
                selections.to_csv(out / f"screening_{h}d.csv", index=False)
                quotes = load_quotes(source / "daily_pv.h5", set(signals.ticker), calendar, "2022-01-01", "2023-12-31")
                report, trades, positions, execution_audit = simulate_cn(signals, quotes, calendar, config, "2022-01-01", "2023-12-31")
                metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
                feasible = (is_feasible(metrics) and execution_audit["liquidation_complete"]
                            and execution_audit["turnover_budget_respected"] and execution_audit["cost_budget_respected"])
                trials.append({"horizon": h, "status": "feasible_in_proxy" if feasible else "not_feasible", "config": asdict(config),
                               **metrics, "execution_audit": execution_audit,
                               "paid_cost_fraction_of_initial_capital": float(trades.total_cost.sum())})
                report.to_csv(out / f"returns_{h}d.csv")
                trades.to_csv(out / f"trades_{h}d.csv", index=False)
                positions.to_csv(out / f"positions_{h}d.csv.gz", index=False)
        for p, expected_hash in hashes.items():
            if file_fingerprint(Path(p)) != expected_hash:
                raise ValueError("Input/code changed: " + p)
        summary = {"research_eligible": pool["eligible_counts"], "strategies": trials, "holdout_evaluated": False,
                   "strategy_window_is_previously_examined": True, "scope": manifest["scope"]}
        feasible = [r for r in trials if r.get("status") == "feasible_in_proxy"]
        selected = max(feasible, key=lambda r: r["sharpe"], default=None)
        write("strategy_selection.json", {"selected_horizon": selected["horizon"] if selected else None,
                                           "status": "selected_in_iteration_proxy" if selected else "no_feasible_strategy" if args.evaluate_strategies else "not_evaluated",
                                           "holdout_evaluated": False})
        write("summary.json", summary)
        write("status.json", {"status": "completed_research"})
        lines = ["# Medium-frequency factor and strategy research", "",
                 f"Research eligibility: {pool['eligible_counts']}. Reserved 2024–2025 not evaluated.", "",
                 "Labels: next-open entry to open after 5/20 holding sessions; research uses fixed nonoverlapping anchors.",
                 "Research 2010–2018 / development 2019–2021; strategy iterations 2022–2023 were previously examined.", "",
                 "| Holding horizon | Status | Net cumulative | Annualized | Max drawdown | Mean daily turnover |", "|---|---|---:|---:|---:|---:|"]
        for trial in trials:
            if "cumulative_return" in trial:
                lines.append(f"| {trial['horizon']}d | {trial['status']} | {trial['cumulative_return']:.2%} | {trial['annualized_return']:.2%} | {trial['max_drawdown']:.2%} | {trial['mean_daily_turnover']:.2%} |")
            else:
                lines.append(f"| {trial['horizon']}d | {trial['status']} | — | — | — | — |")
        lines += ["", "Budget limits include initial formation and terminal wind-down; locked or untradable inventory is never erased.",
                  "Costs, signal definitions, screening rules and source hashes were frozen before strategy iteration.", "", *["- " + line for line in manifest["limitations"]]]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
