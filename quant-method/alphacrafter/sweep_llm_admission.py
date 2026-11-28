#!/usr/bin/env python3
"""Audit relaxation needed for frozen LLM factors, never certify historical PIT."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from alphax.factor_dsl import candidate_from_dict
from alphax.miner_metrics import PAPER_HORIZON_THRESHOLDS
from trader_lite import file_fingerprint

# Coverage and turnover remain unchanged in all policies.
COVERAGE_FLOOR = .90
TURNOVER_CEILING = .40
POLICIES = (
    ("paper_1d_5d", (1, 5), 1., .60),
    ("paper_1d_only", (1,), 1., .60),
    ("half_paper_1d_5d", (1, 5), .5, .55),
    ("quarter_paper_1d_5d", (1, 5), .25, .52),
    ("positive_1d_5d_hit_above_50", (1, 5), 0., .50),
    ("positive_1d_hit_above_50", (1,), 0., .50),
    ("positive_1d_5d_hit_above_49", (1, 5), 0., .49),
)


def gate(row: dict, horizons: tuple[int, ...], fraction: float, hit_floor: float) -> bool:
    direction = row["direction_from_train"]
    if direction not in (-1, 1):
        return False
    for horizon in horizons:
        thresholds = PAPER_HORIZON_THRESHOLDS[horizon]
        for split in ("train", "validation"):
            metric = row["horizons"][str(horizon)][split]
            for key, floor in (("mean_daily_ic", thresholds["ic"]),
                               ("mean_daily_rank_ic", thresholds["rank_ic"]),
                               ("icir", thresholds["icir"]),
                               ("rank_icir", thresholds["rank_icir"])):
                value = direction * metric[key]
                if not np.isfinite(value) or value <= fraction * floor:
                    return False
            hit = metric["rank_ic_hit_ratio"] if direction == 1 else 1 - metric["rank_ic_hit_ratio"]
            if not (np.isfinite(hit) and hit > hit_floor and np.isfinite(metric["coverage"])
                    and metric["coverage"] > COVERAGE_FLOOR and np.isfinite(metric["turnover"])
                    and metric["turnover"] < TURNOVER_CEILING):
                return False
    return True


def build_all_admit_ladder(findings: list[dict], steps: int = 100) -> dict:
    """A monotone path from paper floors to observed worst-case floors (not a fit)."""
    if not findings or steps < 1 or len({r["candidate_id"] for r in findings}) != len(findings):
        raise ValueError("Ladder needs unique candidate IDs and positive steps")
    if any(r["direction_from_train"] not in (-1, 1) for r in findings):
        raise ValueError("Cannot freeze factor direction")
    keys = ("mean_daily_ic", "mean_daily_rank_ic", "icir", "rank_icir", "hit_ratio")
    measured = {str(h): {key: [] for key in keys} for h in (1, 5)}
    for row in findings:
        direction = row["direction_from_train"]
        for h in (1, 5):
            for split in ("train", "validation"):
                m = row["horizons"][str(h)][split]
                if not (np.isfinite(m["coverage"]) and m["coverage"] > COVERAGE_FLOOR
                        and np.isfinite(m["turnover"]) and m["turnover"] < TURNOVER_CEILING):
                    raise ValueError("Not all factors pass fixed coverage and turnover; cannot admit all without changing those gates")
                for key in keys[:-1]:
                    measured[str(h)][key].append(direction * m[key])
                measured[str(h)]["hit_ratio"].append(m["rank_ic_hit_ratio"] if direction > 0 else 1 - m["rank_ic_hit_ratio"])
    if not all(np.isfinite(values).all() for h in measured.values() for values in h.values()):
        raise ValueError("Non-finite metrics cannot be admitted by finite thresholds")
    paper = {str(h): {"mean_daily_ic": PAPER_HORIZON_THRESHOLDS[h]["ic"],
                      "mean_daily_rank_ic": PAPER_HORIZON_THRESHOLDS[h]["rank_ic"],
                      "icir": PAPER_HORIZON_THRESHOLDS[h]["icir"],
                      "rank_icir": PAPER_HORIZON_THRESHOLDS[h]["rank_icir"], "hit_ratio": .60} for h in (1, 5)}
    # Strict > requires a small margin below the observed minimum. This is an
    # empirical envelope, not a generalizable target threshold.
    minimum = {h: {key: float(min(values)) for key, values in metrics.items()} for h, metrics in measured.items()}
    targets = {h: {key: min(paper[h][key], value) - 1e-9 for key, value in metrics.items()}
               for h, metrics in minimum.items()}
    rows = []
    first_entry: dict[str, dict] = {}
    prior: set[str] = set()
    for step in range(steps + 1):
        t = step / steps
        floors = {h: {key: paper[h][key] + t * (targets[h][key] - paper[h][key]) for key in keys} for h in paper}
        ids = []
        for row in findings:
            direction = row["direction_from_train"]
            if all(
                all(direction * row["horizons"][str(h)][split][key] > floors[str(h)][key] for key in keys[:-1])
                and (row["horizons"][str(h)][split]["rank_ic_hit_ratio"] if direction > 0 else
                     1 - row["horizons"][str(h)][split]["rank_ic_hit_ratio"]) > floors[str(h)]["hit_ratio"]
                for h in (1, 5) for split in ("train", "validation")
            ):
                ids.append(row["candidate_id"])
                first_entry.setdefault(row["candidate_id"], {"step": step, "fraction": t})
        if not prior <= set(ids):
            raise ValueError("Relaxation ladder was not monotone")
        prior = set(ids)
        net = [r.get("admission_window_trade_diagnostic", {}).get("net_cumulative") for r in findings if r["candidate_id"] in ids]
        net = [float(x) for x in net if x is not None and np.isfinite(x)]
        rows.append({"step": step, "fraction": t, "floors": floors, "admitted_ids": ids,
                     "same_admission_window_trade_count": len(net),
                     "same_admission_window_profitable_count": sum(x > 0 for x in net),
                     "same_admission_window_median_net": float(np.median(net)) if net else None})
    if len(rows[-1]["admitted_ids"]) != len(findings):
        raise ValueError("Final floors did not admit all factors")
    return {"scope": "posthoc_envelope_not_admission_recommendation", "steps": rows,
            "empirical_minima": minimum, "all_admit_floors": targets,
            "first_entry": first_entry, "coverage_floor": COVERAGE_FLOOR,
            "turnover_ceiling": TURNOVER_CEILING}


def run_sweep(findings: list[dict]) -> list[dict]:
    return [{"policy": name, "horizons": list(horizons), "paper_ic_icir_fraction": fraction,
             "hit_floor": hit, "coverage_floor": COVERAGE_FLOOR,
             "turnover_ceiling": TURNOVER_CEILING,
             "admitted_ids": [row["candidate_id"] for row in findings if gate(row, horizons, fraction, hit)]}
            for name, horizons, fraction, hit in POLICIES]


def retrospective_library(rows: list[dict], candidates: dict[str, dict], policy: dict, cutoff: str) -> dict:
    """Current-time exploratory artifact; never a backdated Miner/Screener snapshot."""
    now = datetime.now(timezone.utc).isoformat()
    factors = []
    for candidate_id in policy["admitted_ids"]:
        candidate = candidate_from_dict(candidates[candidate_id])
        row = next(item for item in rows if item["candidate_id"] == candidate_id)
        factors.append({"candidate_id": candidate_id, "op": candidate.op, "features": list(candidate.features),
                        "expression_spec": candidate.expression_spec, "expression": candidate.expression,
                        "hypothesis": candidate.hypothesis, "direction": row["direction_from_train"],
                        "admission_policy": policy["policy"], "strict_paper_accepted": False,
                        "historical_metric_cutoff": cutoff, "created_at_utc": now, "status": "active"})
    return {"library_version": "threshold-sweep-retrospective-v1", "library_scope": "retrospective_diagnostic",
            "generated_at_utc": now, "historical_metric_cutoff": cutoff,
            "admission_policy": policy, "factors": factors,
            "warning": "Proposals postdate historical cutoff; not PIT, fresh OOS, strict Miner acceptance, or a live-trading library"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    batch = args.batch_dir.resolve(strict=True)
    paths = [batch / name for name in ("status.json", "manifest.json", "findings.json", "candidates.json", "generation_audit.json")]
    if json.loads(paths[0].read_text()).get("status") != "completed_research":
        raise ValueError("Require a completed research batch")
    manifest = json.loads(paths[1].read_text())
    if manifest.get("scope") != "retrospective_research_not_fresh_oos_or_tradable_library":
        raise ValueError("Cannot sweep unlabelled historical results")
    findings = [row for row in json.loads(paths[2].read_text()) if row["origin"] == "frozen_llm"]
    candidates = {row["candidate_id"]: row for row in json.loads(paths[3].read_text())}
    llm_ids = {row["candidate_id"] for row in json.loads(paths[4].read_text())
               if row["selected"] and row["origin"] == "frozen_llm"}
    if not findings or {row["candidate_id"] for row in findings} != llm_ids:
        raise ValueError("Frozen LLM rows missing from batch")
    if {row["candidate_id"] for row in findings} - set(candidates):
        raise ValueError("Missing frozen candidate definitions")
    pinned = {str(p): file_fingerprint(p) for p in [*paths, Path(__file__).resolve()]}
    results = run_sweep(findings)
    ladder = build_all_admit_ladder(findings)
    if results[0]["admitted_ids"] != [r["candidate_id"] for r in findings if r["strict_accepted"]] or ladder["steps"][0]["admitted_ids"] != results[0]["admitted_ids"]:
        raise ValueError("Paper gate mismatch: do not publish relaxation results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"batch_inputs": pinned,
        "scope": "retrospective_threshold_sensitivity_only", "historical_metric_cutoff": manifest["as_of"]}, indent=2) + "\n")
    (args.output_dir / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "ladder.json").write_text(json.dumps(ladder, indent=2, allow_nan=False) + "\n")
    # Current-time retrospective research libraries only; no PIT snapshots or strict registry.
    for name in ("positive_1d_hit_above_50", "positive_1d_5d_hit_above_49"):
        policy = next(p for p in results if p["policy"] == name)
        payload = retrospective_library(findings, candidates, policy, manifest["as_of"])
        (args.output_dir / f"library_{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    final_step = ladder["steps"][-1]
    all_policy = {"policy": "posthoc_all_admit_envelope_not_effective_gate", "horizons": [1, 5],
                  "thresholds": ladder["all_admit_floors"], "hit_floor_per_horizon": {h: v["hit_ratio"] for h, v in ladder["all_admit_floors"].items()},
                  "coverage_floor": COVERAGE_FLOOR, "turnover_ceiling": TURNOVER_CEILING,
                  "admitted_ids": final_step["admitted_ids"]}
    (args.output_dir / "library_all_admit_retrospective.json").write_text(
        json.dumps(retrospective_library(findings, candidates, all_policy, manifest["as_of"]), indent=2) + "\n")
    lines = ["# Frozen LLM admission threshold sweep (retrospective only)", "",
             "12 or fewer historical LLM candidates, direction fixed by pre-2022 training; 2022 admission already viewed.",
             "All policies retain Coverage >90%, Turnover <40%, require BOTH training and admission to pass; strict inequalities.",
             "Fractions scale the original per-horizon IC/Rank IC and ICIR/Rank ICIR floors; fraction 0 means strictly positive, NOT allowing zero.",
             "The 1-day policies explicitly REMOVE the 5-day requirement; the 49% hit policy admits near-coin-flip sign agreement and is for sensitivity analysis ONLY.",
             "Historical 2022 trade diagnostics are not used for the gate. Libraries are tagged retrospective, generated now, never backdated PIT snapshots.", "",
             "| Policy | horizons | IC/ICIR floor | hit floor | passes | IDs |", "|---|---|---:|---:|---:|---|"]
    for p in results:
        lines.append(f"| {p['policy']} | {p['horizons']} | {p['paper_ic_icir_fraction']:.0%} of paper | >{p['hit_floor']:.0%} | {len(p['admitted_ids'])} | {', '.join(p['admitted_ids']) or '-'} |")
    for p in results:
        if not p["admitted_ids"]:
            continue
        lines.append(f"\n## {p['policy']} admitted metrics and **same-window** simulation (not OOS)\n")
        for candidate_id in p["admitted_ids"]:
            r = next(item for item in findings if item["candidate_id"] == candidate_id)
            d = r["direction_from_train"]
            one = r["horizons"]["1"]
            trade = r["admission_window_trade_diagnostic"]
            net = f"{trade['net_cumulative']:.2%}" if trade["status"] == "simulated" else trade["status"]
            lines.append(f"- `{candidate_id}`: 1d train/validation directional Rank IC "
                         f"{d*one['train']['mean_daily_rank_ic']:.5f}/{d*one['validation']['mean_daily_rank_ic']:.5f}; "
                         f"2022 admission-window 10%-exposure net {net} (not a selection criterion).")
    lines += ["", "## Continuous 1d+5d relaxation to ALL factors (post-hoc, NOT an effective standard)",
              "", "101 steps interpolate ALL five per-horizon paper floors to the observed worst metric minus 1e-9; coverage >90% and turnover <40% remain unchanged. This forces admission and therefore cannot validate predictive strength.",
              "Full thresholds and admitted IDs for every step: `ladder.json`; final current-time-only research list: `library_all_admit_retrospective.json`.",
              "2022 net trade counts below are SAME admission window, not independent validation.", "",
              "| relaxed fraction | admitted | 2022 profitable / simulated | median 2022 net |",
              "|---:|---:|---:|---:|"]
    for step in ladder["steps"]:
        if step["step"] in (0, 25, 50, 75, 100) or (step["step"] > 0 and
                len(step["admitted_ids"]) != len(ladder["steps"][step["step"] - 1]["admitted_ids"])):
            median = step["same_admission_window_median_net"]
            lines.append(f"| {step['fraction']:.0%} | {len(step['admitted_ids'])} | {step['same_admission_window_profitable_count']} / {step['same_admission_window_trade_count']} | {median:.2%} |" if median is not None else
                         f"| {step['fraction']:.0%} | {len(step['admitted_ids'])} | 0 / 0 | — |")
    lines += ["", "Final threshold floors (strict >) by horizon:", "",
              "| horizon | IC | Rank IC | ICIR | Rank ICIR | directional hit |",
              "|---:|---:|---:|---:|---:|---:|"]
    for h, floors in ladder["all_admit_floors"].items():
        lines.append(f"| {h}d | {floors['mean_daily_ic']:.6f} | {floors['mean_daily_rank_ic']:.6f} | {floors['icir']:.5f} | {floors['rank_icir']:.5f} | {floors['hit_ratio']:.4%} |")
    lines += ["", "First entry step for each factor (lower step = survives a stricter rule):"]
    for candidate_id, entry in sorted(ladder["first_entry"].items(), key=lambda item: (item[1]["step"], item[0])):
        lines.append(f"- step {entry['step']}/100: `{candidate_id}`")
    lines.append("\nNo new strict library or paper-accepted factor is created. The 2023 demo already used the 1d-only factor; repeatedly testing 2023+ here would be reuse, not fresh validation.\n")
    (args.output_dir / "report.md").write_text("\n".join(lines))
    for path, expected in pinned.items():
        if file_fingerprint(Path(path)) != expected:
            raise ValueError(f"Source changed during sweep: {path}")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_research"}) + "\n")
    print("\n".join(f"{r['policy']}: {len(r['admitted_ids'])} {r['admitted_ids']}" for r in results))
    print(f"all-admit ladder: {len(final_step['admitted_ids'])}/{len(findings)}; floors: {json.dumps(ladder['all_admit_floors'], sort_keys=True)}")


if __name__ == "__main__":
    main()
