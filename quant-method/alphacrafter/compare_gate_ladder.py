#!/usr/bin/env python3
"""Compare distinct relaxed LLM library sets under ONE fixed 2023 local simulator.

Retrospective research only: LLM proposals postdate historical cutoff, 2023 was
already examined, and threshold path is fitted to 2022 admission metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from trader_lite import TraderConfig, file_fingerprint, load_inputs, simulate, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "machine-learn"
CONFIG = TraderConfig(beta=.1, gamma=1., nlong=10, nshort=0)
COST_RATE = .0001
INITIAL_CASH = 1_000_000


def distinct_sets(steps: list[dict]) -> list[dict]:
    """All 101 thresholds are represented; identical accepted sets need one run."""
    if not steps or [step["step"] for step in steps] != list(range(len(steps))):
        raise ValueError("Ladder steps must be consecutive starting from zero")
    result = []
    seen = set()
    prior = set()
    for step in steps:
        names = step["admitted_ids"]
        if len(names) != len(set(names)) or not prior <= set(names):
            raise ValueError("Non-monotone or duplicate admission set")
        prior = set(names)
        key = frozenset(names)
        if key not in seen:
            result.append({"label": f"step_{step['step']:03d}", "first_step": step["step"], "first_fraction": step["fraction"],
                           "admitted_ids": list(names), "step_floors": step["floors"]})
            seen.add(key)
    return result


def status_summary(rows: list[dict]) -> dict:
    if not rows or not all(row["status"] == "simulated" for row in rows):
        raise ValueError("All comparison groups must have valid simulation results")
    best = max(rows, key=lambda row: (row["metrics"]["cumulative_return"], -row["first_step"]))
    return {"best_variant": best.get("label", f"step_{best['first_step']:03d}"),
            "best_first_step": best["first_step"], "best_admitted_count": len(best["admitted_ids"]),
            "best_net_return": best["metrics"]["cumulative_return"],
            "warning": "Best is post-hoc on already-viewed 2023, NOT an effective standard or fresh OOS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/qlib/alphacrafter_alpha158_curated_v1/features.csv.gz")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/processed/sp500_xnys_xcal_4_13_3.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2023-01-03")
    parser.add_argument("--end", default="2023-12-29")
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        parser.error("Require ordered signal window")
    sweep = args.sweep_dir.resolve(strict=True)
    if json.loads((sweep / "status.json").read_text()).get("status") != "completed_research":
        raise ValueError("Incomplete relaxation sweep")
    manifest = json.loads((sweep / "manifest.json").read_text())
    if manifest.get("scope") != "retrospective_threshold_sensitivity_only":
        raise ValueError("This comparison requires a retrospective-labelled sweep")
    ladder = json.loads((sweep / "ladder.json").read_text())
    variants = distinct_sets(ladder["steps"])
    if variants[-1]["admitted_ids"] != list(ladder["steps"][-1]["admitted_ids"]):
        raise ValueError("No all-admit variant")
    all_library = json.loads((sweep / "library_all_admit_retrospective.json").read_text())
    if all_library.get("library_scope") != "retrospective_diagnostic" or not all_library.get("factors"):
        raise ValueError("Refuse to use an unmarked or empty library")
    factor_by_id = {row["candidate_id"]: row for row in all_library["factors"]}
    if len(factor_by_id) != len(all_library["factors"]) or set(variants[-1]["admitted_ids"]) != set(factor_by_id):
        raise ValueError("All-admit factor IDs differ from ladder")
    # The 1-day-only rule is NOT on the monotone 1d+5d ladder. Include it
    # separately so the comparison covers the user's earlier demo policy.
    one_day_path = sweep / "library_positive_1d_hit_above_50.json"
    one_day = json.loads(one_day_path.read_text())
    one_day_ids = [row["candidate_id"] for row in one_day["factors"]]
    if (one_day.get("library_scope") != "retrospective_diagnostic" or not one_day_ids
            or not set(one_day_ids) <= set(factor_by_id) or len(one_day_ids) != len(set(one_day_ids))
            or any(set(v["admitted_ids"]) == set(one_day_ids) for v in variants)):
        raise ValueError("Invalid or duplicate 1d-only research library")
    variants.append({"label": "policy_positive_1d_only", "first_step": -1, "first_fraction": None,
                     "admitted_ids": one_day_ids, "step_floors": one_day["admission_policy"]})
    args.input, args.calendar = args.input.resolve(strict=True), args.calendar.resolve(strict=True)
    sources = [sweep / name for name in ("status.json", "manifest.json", "ladder.json", "library_all_admit_retrospective.json", "library_positive_1d_hit_above_50.json")]
    sources += [args.input, args.calendar, HERE / "trader_lite.py", HERE / "screener_lite.py", HERE / "compare_gate_ladder.py"]
    pinned = {str(p): file_fingerprint(p) for p in sources}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"scope": "retrospective_2023_already_viewed_not_oos",
        "signal_dates": [args.start, args.end], "config": CONFIG.__dict__,
        "cost_rate_per_leg": COST_RATE, "initial_cash": INITIAL_CASH, "step_count": len(ladder["steps"]),
        "distinct_admission_sets": len(variants), "source_fingerprints": pinned,
        "warning": "LLM factors generated after historical cutoff; PIT universe biased; historical best cannot set an effective gate"}, indent=2) + "\n")
    rows = []
    try:
        for variant in variants:
            step = variant["first_step"]
            directory = args.output_dir / variant["label"]
            directory.mkdir()
            payload = {**all_library, "factors": [factor_by_id[cid] for cid in variant["admitted_ids"]],
                       "admission_policy": {"policy": variant["label"] + "_not_certified",
                                            "step": step, "floors": variant["step_floors"]}}
            library = directory / "library_retrospective.json"
            library.write_text(json.dumps(payload, indent=2) + "\n")
            command = [sys.executable, str(HERE / "screener_lite.py"), "--input", str(args.input),
                       "--calendar", str(args.calendar), "--require-explicit-label-dates",
                       "--allow-retrospective-library", "--factor-library", str(library),
                       "--signal-start", args.start, "--as-of", args.end,
                       "--output-dir", str(directory / "screener")]
            with (directory / "screener.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            screen = directory / "screener"
            screen_registry = json.loads((screen / "registry.json").read_text())
            if screen_registry.get("library_scope") != "retrospective_diagnostic" or screen_registry.get("invalid_factors"):
                raise ValueError("Screener did not accept and label every retrospective factor")
            predictions = screen / "predictions.csv"
            frame = load_inputs(predictions, args.input, as_of=args.end,
                                calendar_path=args.calendar, require_explicit_label_dates=True)
            report, positions = simulate(frame, CONFIG, args.start, args.end, COST_RATE)
            selection = pd.read_csv(screen / "selection.csv")
            if not set(selection["candidate_id"]) <= set(variant["admitted_ids"]):
                raise ValueError("Screener selected an unadmitted factor")
            report.to_csv(directory / "trades_daily.csv")
            positions.to_csv(directory / "positions.csv", index=False)
            metrics = summarize(report, INITIAL_CASH, .0381)
            rows.append({"status": "simulated", "label": variant["label"], "first_step": step, "first_fraction": variant["first_fraction"],
                         "admitted_ids": variant["admitted_ids"], "selected_factor_ids": sorted(set(selection["candidate_id"])),
                         "selected_factor_days": int(selection["date"].nunique()),
                         "invested_days": int((report["selected_assets"] > 0).sum()),
                         "position_rows": len(positions), "metrics": metrics})
            (args.output_dir / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        if len({(r["metrics"]["start"], r["metrics"]["end"], r["metrics"]["trading_days"]) for r in rows}) != 1:
            raise ValueError("Comparison variants have different evaluation windows")
        if not rows[0]["invested_days"] == 0 or not rows[-2]["admitted_ids"]:
            raise ValueError("Empty/full ladder endpoints do not reconcile")
        summary = status_summary(rows)
        for path, expected in pinned.items():
            if file_fingerprint(Path(path)) != expected:
                raise ValueError(f"Source changed during comparison: {path}")
        (args.output_dir / "results.json").write_text(json.dumps({"scope": "retrospective_2023_already_viewed_not_oos",
            "summary": summary, "distinct_sets": rows}, indent=2, allow_nan=False) + "\n")
        lines = ["# 2023 historical comparison of the entire all-admit threshold ladder",
                 "", "NOT fresh OOS: LLM candidates were generated after 2023; 2023 results already viewed. Thresholds fitted to 2022; historical winner is post-hoc.",
                 f"Each distinct factor set runs the same Screener and daily flat-to-cash Trader: beta=10%, long 10, T+1 open-to-close, 1bp per leg, $1m initial cash. All {len(ladder['steps'])} ladder steps represented by {len(rows)-1} distinct sets, PLUS the separate 1-day-only policy used in the earlier execution demo.",
                 "Screener still uses its fixed trailing Rank IC suitability rule; library admission does NOT guarantee selection or an order.",
                 "", "| policy / first ladder step | admitted | selected days | invested days | net cumulative | max drawdown | Sharpe |",
                 "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            m = r["metrics"]
            lines.append(f"| {r['label']} | {len(r['admitted_ids'])} | {r['selected_factor_days']} | {r['invested_days']} | {m['cumulative_return']:.2%} | {m['max_drawdown']:.2%} | {m['sharpe'] if m['sharpe'] is not None else 'undefined'} |")
        lines += ["", f"Highest HISTORICAL net: {summary['best_variant']}, {summary['best_admitted_count']} factors, {summary['best_net_return']:.2%} (not an admission recommendation).",
                  "", "Each policy's exact factor IDs, positions and daily P&L: `results.json` and `step_NNN/` or `policy_positive_1d_only/`; shared-set steps in `ladder.json`. No broker fills, validated historical PIT universe, or fresh sample."]
        (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
        (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_research"}) + "\n")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        (args.output_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
