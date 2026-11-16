"""Create the research pool and an independent strategy-validation plan from frozen Miner metrics."""
import argparse
import json
from pathlib import Path

from admission import build_research_pool, strategy_plan
from trader_lite import file_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--temporal-audit", type=Path, required=True)
    parser.add_argument("--market", choices=("cn", "us"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy-validation-start", default="2022-01-01")
    parser.add_argument("--strategy-validation-end", default="2023-12-31")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory")
    inputs = [args.metrics, args.candidates, args.temporal_audit, Path(__file__), Path(__file__).with_name("admission.py")]
    fingerprints = {str(p.resolve()): file_fingerprint(p) for p in inputs}
    results, definitions, audit = [json.loads(p.read_text()) for p in inputs[:3]]
    pool, assessments = build_research_pool(results, definitions, audit, args.market)
    plan = strategy_plan(pool, args.strategy_validation_start, args.strategy_validation_end,
                         args.holdout_start, args.holdout_end)
    for path, expected in fingerprints.items():
        if file_fingerprint(Path(path)) != expected:
            raise ValueError("Research input changed: " + path)
    args.output_dir.mkdir(parents=True)

    def write(name, data):
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    write("research_pool.json", pool)
    write("research_assessments.json", assessments)
    plan["research_pool_fingerprint"] = file_fingerprint(args.output_dir / "research_pool.json")
    write("strategy_validation_plan.json", plan)
    write("manifest.json", {"scope": "two_layer_admission_research_only", "sources": fingerprints,
                            "formal_validation_run": False, "holdout_metrics_read": False})
    write("status.json", {"status": "completed_research", "eligible_counts": pool["eligible_counts"],
                          "formal_strategy_status": plan["status"], "strategy_admitted_count": 0})
    lines = ["# Research pool and formal strategy admission", "",
             f"Research eligible by horizon: {pool['eligible_counts']}. Formal strategies admitted: 0 (validation not run).", "",
             "| Horizon | Candidate | Training direction | Development Rank IC | Development Rank ICIR | Development hit |",
             "|---|---|---:|---:|---:|---:|"]
    for horizon, entries in pool["candidates_by_horizon"].items():
        for entry in entries:
            m, d = entry["metrics"]["validation"], entry["direction"]
            hit = m["rank_ic_hit_ratio"] if d > 0 else 1 - m["rank_ic_hit_ratio"]
            lines.append(f"| {horizon}d | {entry['candidate_id']} | {d} | {d*m['mean_daily_rank_ic']:.5f} | {d*m['rank_icir']:.3f} | {hit:.1%} |")
    lines += ["", "Pool thresholds were chosen after examining prior development results; eligibility is exploratory, not independent validation.",
              "Strategy definitions must be frozen before evaluating the independent validation window. No original joint single-factor gate is required at portfolio validation.",
              "Execution requirement: " + plan["execution_requirement"],
              "No trading factor library, predictions or historical snapshots are exported."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"eligible_counts": pool["eligible_counts"], "strategy_status": plan["status"]}))


if __name__ == "__main__":
    main()
