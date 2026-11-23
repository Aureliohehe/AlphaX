#!/usr/bin/env python3
"""Fixed-budget exploratory factor batch: frozen LLM proposals + interpretable ASTs.

2022 has already been examined in this repository. This is retrospective research,
not a fresh out-of-sample result or a source of paper-qualified live factors.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from alphax.factor_dsl import candidate_from_dict, evaluate_candidate_score, validate_candidate
from miner_lite import PAPER_HORIZON_THRESHOLDS, PAPER_MIN_HIT_DEVIATION
from temporal import read_calendar
from trader_lite import TraderConfig, file_fingerprint, simulate, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "machine-learn"
COST_RATE = .0001
CONFIG = TraderConfig(beta=.1, gamma=1., nlong=10, nshort=0)
FAMILIES = {
    "trend": ["alpha158_roc10", "alpha158_roc30", "alpha158_ma10", "alpha158_ma30"],
    "range": ["alpha158_rsv20", "alpha158_kmid2", "alpha158_kup2", "alpha158_klow2"],
    "risk": ["alpha158_std10", "alpha158_std30", "alpha158_max20", "alpha158_min20"],
    "participation": ["alpha158_corr20", "alpha158_cntd20", "alpha158_sumd20"],
}


def canonical(node: dict) -> str:
    """Canonicalize structural duplicates only; do not assume correlated factors equal."""
    op = node["op"]
    if op == "feature":
        key = [op, node["feature"]]
    elif op == "ts_corr":
        key = [op, sorted(node["features"]), node["window"]]
    elif op.startswith("ts_"):
        key = [op, node["feature"], node["window"], node.get("quantile")]
    elif "input" in node:
        key = [op, canonical(node["input"])]
    else:
        children = [canonical(child) for child in node["inputs"]]
        key = [op, sorted(children) if op in ("sum", "product") else children]
    return json.dumps(key, sort_keys=True)


def generate(frozen_llm: list[dict], available: set[str], budget: int) -> tuple[list[dict], list[dict]]:
    if budget < 1:
        raise ValueError("Candidate budget must be positive")
    proposals: list[tuple[str, dict]] = [("frozen_llm", row) for row in frozen_llm]
    for family, fields in FAMILIES.items():
        for field in fields:
            proposals.append(("raw_" + family, {
                "candidate_id": "batch_raw_" + field, "expression": {"op": "feature", "feature": field},
                "hypothesis": f"{family} level of {field} may forecast next-session cross-sectional returns",
            }))
    for family, fields in FAMILIES.items():
        for field in fields[:2]:
            for op in ("ts_zscore", "ts_delta"):
                proposals.append(("temporal_" + family, {
                    "candidate_id": f"batch_{op}_{field}_20", "expression": {"op": op, "feature": field, "window": 20},
                    "hypothesis": f"Within-stock 20-session {op} of {field} may detect {family} change",
                }))
    for family, fields in FAMILIES.items():
        left, right = fields[:2]
        proposals.append(("relative_" + family, {
            "candidate_id": f"batch_spread_{left}_{right}",
            "expression": {"op": "difference", "inputs": [{"op": "feature", "feature": left}, {"op": "feature", "feature": right}]},
            "hypothesis": f"Relative {family} signal between {left} and {right} may outperform either alone",
        }))
    accepted, audit, seen, ids = [], [], {}, set()
    for origin, proposal in proposals:
        candidate = candidate_from_dict(proposal)
        validate_candidate(candidate, available)
        if candidate.candidate_id in ids:
            raise ValueError(f"Duplicate candidate ID: {candidate.candidate_id}")
        ids.add(candidate.candidate_id)
        # Legacy proposals use the same safe AST representation for deduplication.
        spec = candidate.expression_spec or (
            {"op": "feature", "feature": candidate.features[0]} if candidate.op == "feature"
            else {"op": candidate.op, "inputs": [{"op": "feature", "feature": field} for field in candidate.features]}
        )
        fingerprint = canonical(spec)
        duplicate_of = seen.get(fingerprint)
        if duplicate_of is None:
            seen[fingerprint] = candidate.candidate_id
        selected = duplicate_of is None and len(accepted) < budget
        audit.append({"candidate_id": candidate.candidate_id, "origin": origin,
                      "hypothesis": candidate.hypothesis, "expression": candidate.expression,
                      "duplicate_of": duplicate_of, "selected": selected})
        if selected:
            accepted.append({"candidate_id": candidate.candidate_id, "features": list(candidate.features),
                             "expression": candidate.expression_spec or spec, "hypothesis": candidate.hypothesis})
    return accepted, audit


def failure_reasons(metric: dict, direction: int, horizon: int) -> list[str]:
    floor = PAPER_HORIZON_THRESHOLDS[horizon]
    checks = {key: direction * metric[key] > floor[threshold] for key, threshold in
              (("mean_daily_ic", "ic"), ("mean_daily_rank_ic", "rank_ic"),
               ("icir", "icir"), ("rank_icir", "rank_icir"))}
    hit = metric["rank_ic_hit_ratio"] if direction > 0 else 1 - metric["rank_ic_hit_ratio"]
    checks.update(hit_ratio=hit > .5 + PAPER_MIN_HIT_DEVIATION,
                  coverage=metric["coverage"] > .90, turnover=metric["turnover"] < .40)
    return [name for name, passed in checks.items() if not passed]


def diagnostic_trades(input_path: Path, calendar_path: Path, candidate_rows: list[dict],
                      metrics: list[dict], start: str, end: str) -> dict[str, dict]:
    """Fixed-direction 10%-exposure trade diagnostics in the *admission* window.

    Never remove missing-return names before selection; simulate() fails closed.
    No trading result here may be used to select/relabel the original Miner gate.
    """
    candidates = {row["candidate_id"]: candidate_from_dict(row) for row in candidate_rows}
    features = sorted({field for candidate in candidates.values() for field in candidate.features})
    columns = ["date", "ticker", "target_open_close", "label_end_1d", *features]
    panel = pd.read_csv(input_path, usecols=columns, parse_dates=["date", "label_end_1d"])
    panel = panel[panel["date"].between(pd.Timestamp(start) - pd.Timedelta(days=500), pd.Timestamp(end))].copy()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    calendar = read_calendar(calendar_path)
    next_date = dict(zip(calendar[:-1], calendar[1:]))
    panel["execution_date"] = panel["date"].map(next_date)
    panel.loc[panel["label_end_1d"] != panel["execution_date"], "target_open_close"] = np.nan
    result = {}
    for row in metrics:
        candidate = candidates[row["candidate_id"]]
        score = evaluate_candidate_score(candidate, panel)
        dates = panel["date"].between(start, end)
        signal = panel.loc[dates, ["date", "ticker", "target_open_close", "execution_date"]].copy()
        signal["prediction"] = row["direction_from_train"] * score[dates].to_numpy()
        try:
            report, positions = simulate(signal, CONFIG, start, end, COST_RATE)
            result[candidate.candidate_id] = {
                "status": "simulated", "invested_days": int((report["selected_assets"] > 0).sum()),
                "positions": len(positions), "gross_cumulative": float((1 + report["gross_return"]).prod() - 1),
                "net_cumulative": summarize(report, 1_000_000)["cumulative_return"],
            }
        except ValueError as exc:
            result[candidate.candidate_id] = {"status": "failed_closed", "reason": str(exc)}
    return result


def remove_tradable_artifacts(miner_dir: Path) -> None:
    """Historical LLM proposals must never leave a backdated PIT library behind."""
    if not miner_dir.exists():
        return
    for name in ("factor_library.json", "registry.json", "accepted_factors.json",
                 "predictions.csv", "diagnostic_predictions.csv"):
        (miner_dir / name).unlink(missing_ok=True)
    if (miner_dir / "snapshots").exists():
        shutil.rmtree(miner_dir / "snapshots")
    (miner_dir / "RETROSPECTIVE_ONLY.txt").write_text(
        "Historical Miner metrics only. Proposals were generated after the cutoff; "
        "no certified factor library, registry, PIT snapshot or signals are exported.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/qlib/alphacrafter_alpha158_curated_v1/features.csv.gz")
    parser.add_argument("--calendar", type=Path, default=ROOT / "data/processed/sp500_xnys_xcal_4_13_3.csv")
    parser.add_argument("--llm-candidates", type=Path, default=ROOT / "data/qlib/alphacrafter_agent_2022_strict_v2/candidates.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--as-of", default="2022-12-30")
    parser.add_argument("--train-start", default="2016-01-04")
    parser.add_argument("--train-end", default="2021-12-30")
    parser.add_argument("--validation-start", default="2022-01-03")
    parser.add_argument("--validation-end", default="2022-12-30")
    parser.add_argument("--skip-trades", action="store_true", help="Skip slower per-factor admission-window trade diagnostics")
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    if not (pd.Timestamp(args.train_start) <= pd.Timestamp(args.train_end) < pd.Timestamp(args.validation_start)
            <= pd.Timestamp(args.validation_end) <= pd.Timestamp(args.as_of)):
        parser.error("Train/validation windows must be ordered before as-of")
    args.input, args.calendar, args.llm_candidates = (p.resolve(strict=True) for p in (args.input, args.calendar, args.llm_candidates))
    # The schema is checked against the dataset, not a list invented by the generator.
    available = set(pd.read_csv(args.input, nrows=0).columns) - {"date", "ticker", "target_1d", "target_5d", "target_open_close", "target_open_close_5d", "label_end_1d", "label_end_5d"}
    frozen_llm = json.loads(args.llm_candidates.read_text())
    if not isinstance(frozen_llm, list):
        raise ValueError("Frozen LLM candidate file must be a JSON list")
    candidates, audit = generate(frozen_llm, available, args.budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (args.output_dir / "generation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    paths = [
        args.input, args.calendar, args.llm_candidates, HERE / "miner_lite.py",
        HERE / "alphax/factor_dsl.py", HERE / "trader_lite.py", HERE / "run_candidate_batch.py",
        args.output_dir / "candidates.json",
    ]
    fingerprints = {str(p): file_fingerprint(p) for p in paths}
    manifest = {"scope": "retrospective_research_not_fresh_oos_or_tradable_library",
                "candidate_budget": args.budget, "selected_candidates": len(candidates),
                "training": [args.train_start, args.train_end], "admission": [args.validation_start, args.validation_end],
                "as_of": args.as_of, "label_mode": "execution", "trade_config": CONFIG.__dict__,
                "cost_rate_per_leg": COST_RATE, "skip_trades": args.skip_trades, "files": fingerprints,
                "limitations": "2022 already examined; LLM proposed after historical cutoff; current constituent snapshot biased"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        cmd = [sys.executable, str(HERE / "miner_lite.py"), "--input", str(args.input),
               "--calendar", str(args.calendar), "--require-explicit-label-dates",
               "--candidate-file", str(args.output_dir / "candidates.json"),
               "--output-dir", str(args.output_dir / "miner"), "--as-of", args.as_of,
               "--train-start", args.train_start, "--train-end", args.train_end,
               "--validation-start", args.validation_start, "--validation-end", args.validation_end,
               "--backtest-start", str((pd.Timestamp(args.as_of) + pd.Timedelta(days=1)).date()),
               "--backtest-end", str((pd.Timestamp(args.as_of) + pd.Timedelta(days=1)).date())]
        try:
            with (args.output_dir / "miner.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            remove_tradable_artifacts(args.output_dir / "miner")
        rows = json.loads((args.output_dir / "miner/candidate_metrics.json").read_text())
        if len(rows) != len(candidates) or {r["candidate_id"] for r in rows} != {r["candidate_id"] for r in candidates}:
            raise ValueError("Miner evaluated a different candidate set")
        origin = {r["candidate_id"]: r for r in audit}
        trades = {} if args.skip_trades else diagnostic_trades(args.input, args.calendar, candidates, rows, args.validation_start, args.validation_end)
        findings = []
        for row in rows:
            failures = {f"{split}_{h}d": failure_reasons(row["horizons"][str(h)][split], row["direction_from_train"], h)
                        for split in ("train", "validation") for h in (1, 5)}
            if {key: not value for key, value in failures.items()} != row["gate_results"]:
                raise ValueError(f"Gate diagnosis disagrees with Miner for {row['candidate_id']}")
            findings.append({"candidate_id": row["candidate_id"], "origin": origin[row["candidate_id"]]["origin"],
                             "hypothesis": origin[row["candidate_id"]]["hypothesis"],
                             "expression": origin[row["candidate_id"]]["expression"],
                             "direction_from_train": row["direction_from_train"], "strict_accepted": row["accepted"],
                             "horizons": row["horizons"], "failed_checks": failures,
                             "admission_window_trade_diagnostic": trades.get(row["candidate_id"], {"status": "skipped"})})
        for p, expected in fingerprints.items():
            if file_fingerprint(Path(p)) != expected:
                raise ValueError(f"Source changed during run: {p}")
        (args.output_dir / "findings.json").write_text(json.dumps(findings, indent=2, allow_nan=True) + "\n")
        counts = Counter(reason for r in findings for failures in r["failed_checks"].values() for reason in failures)
        summary = {"scope": manifest["scope"], "candidates": len(candidates),
                   "llm_candidates": sum(r["origin"] == "frozen_llm" for r in findings),
                   "duplicates": [r for r in audit if r["duplicate_of"]],
                   "strict_accepted": [r["candidate_id"] for r in findings if r["strict_accepted"]],
                   "failed_check_counts": dict(counts),
                   "trade_diagnostics": dict(Counter(r["admission_window_trade_diagnostic"]["status"] for r in findings))}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines = ["# Retrospective candidate research (NOT fresh OOS)", "",
                 f"Evaluated {len(findings)} factors; strict accepted {len(summary['strict_accepted'])}. Trade results (if present) are in the **same 2022 admission window** and must not be used as held-out returns.",
                 "Historical constituent PIT and LLM proposal timestamps are not certified. No failed candidate is placed in the strict factor library.", "",
                 "| ID | Origin | train 1d Rank IC | validation 1d Rank IC | failed gates | admission trade net |", "|---|---|---:|---:|---|---:|"]
        for row in findings:
            gates = ", ".join(k for k, v in row["failed_checks"].items() if v) or "none"
            trade = row["admission_window_trade_diagnostic"]
            net = f"{trade['net_cumulative']:.2%}" if trade["status"] == "simulated" else trade["status"]
            lines.append(f"| {row['candidate_id']} | {row['origin']} | {row['horizons']['1']['train']['mean_daily_rank_ic']:.4f} | {row['horizons']['1']['validation']['mean_daily_rank_ic']:.4f} | {gates} | {net} |")
        lines += ["", "Detailed hypotheses, each 1d/5d metric, per-gate failures, and trade failures: `findings.json`; duplicate proposals: `generation_audit.json`."]
        (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
        feedback = ["# Candidate research feedback (not a historical PIT LLM prompt)", "",
                    "These proposals and 2022 scores were assembled retrospectively. Never backdate this feedback or claim that 2022 is fresh OOS.",
                    "The Miner direction is fixed on pre-2022 training Rank IC; the unchanged strict gate requires both 1d and 5d train/admission passes.",
                    "Failure counts over all four splits/horizons: " + json.dumps(dict(counts), sort_keys=True),
                    "Develop mechanisms that address the weak train/admission predictive strength and directional stability; do not optimize 2023+ returns, copy prior candidates, or just relax thresholds.",
                    "Data is a current-snapshot constituent universe, so PIT membership must be fixed before any performance claim.",
                    "See `findings.json` for full metrics and `generation_audit.json` for prior proposals; propose new safe ASTs and hypotheses, not predicted winners."]
        (args.output_dir / "feedback.md").write_text("\n\n".join(feedback) + "\n")
        (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_research"}) + "\n")
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        (args.output_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
