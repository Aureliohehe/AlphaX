"""Read completed weekly research as evidence for a later seeded Miner run."""
import json
import math
from pathlib import Path

from temporal import session_date
from trader_lite import file_fingerprint


METRICS = ("cumulative_return", "annualized_return", "sharpe", "max_drawdown")


def finite_metric(row, key):
    value = row.get(key)
    return value if isinstance(value, (float, int)) and math.isfinite(value) else None


def describe_trade(trial, baseline, thresholds):
    breakdown = {}
    for name, result, core in (("standard", trial, baseline),
                               ("double_cost", trial.get("double_cost", {}), baseline.get("double_cost", {}))):
        metrics = {key: finite_metric(result, key) for key in METRICS}
        control = {key: finite_metric(core, key) for key in METRICS}
        breakdown[name] = {
            "metrics": metrics,
            "delta_vs_core": {key: metrics[key] - control[key] if metrics[key] is not None and control[key] is not None
                              else None for key in METRICS},
            "failed_thresholds": [key for key in thresholds
                                  if metrics[key] is None or metrics[key] <= thresholds[key]],
            "failed_execution_checks": [key for key in ("liquidation_complete", "turnover_budget_respected",
                                                        "cost_budget_respected")
                                        if not result.get("execution_audit", {}).get(key)],
        }
    return {"status": trial["status"], "cost_scenarios": breakdown}


def load_weekly_feedback(directory: Path, as_of, control_ids):
    names = ("status.json", "strategy_selection.json", "summary.json",
             "combination_assessments.json", "definitions_frozen.json")
    paths = {name: directory / name for name in names}
    fingerprints = {name: file_fingerprint(path) for name, path in paths.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    selection = data["strategy_selection.json"]
    if (data["status.json"].get("status") != "completed_research"
            or selection.get("status") not in {"selected_in_iteration_proxy", "no_feasible_strategy"}
            or selection.get("holdout_evaluated") is not False
            or data["summary.json"].get("holdout_evaluated") is not False):
        raise ValueError("Expected completed research-window evaluation without holdout results")
    window = selection["selection_window"]
    cutoff = session_date(as_of)
    if (len(window) != 2 or session_date(window[0]) >= session_date(window[1])
            or session_date(window[1]) > cutoff or session_date(selection["frozen_as_of"]) > cutoff):
        raise ValueError("Prior validation must end on or before the next Miner cutoff")
    trials = {row.get("strategy_id"): row for row in data["summary.json"]["strategies"] if row["horizon"] == 5}
    baseline = trials.get("core4", {})
    thresholds = selection["risk_thresholds"]["weekly_combination_both_costs"]
    definitions = {row["candidate_id"]: row for row in data["definitions_frozen.json"]}
    by_control = {cid: [] for cid in control_ids}
    previous_expressions = set()
    for row in data["combination_assessments.json"]:
        strategy = row["strategy_id"]
        if row["horizon"] != 5 or not strategy.startswith("core4_replace_"):
            continue
        candidate_id = strategy.removeprefix("core4_replace_")
        definition = definitions[candidate_id]
        parent = definition["control_id"]
        if parent not in by_control:
            continue
        for split in row.get("windows", {}).values():
            if session_date(split[1]) > cutoff:
                raise ValueError("Prior combination metrics extend beyond the next Miner cutoff")
        expression = definition["expression_text"]
        previous_expressions.add(expression)
        by_control[parent].append({
            "candidate_id": candidate_id, "expression": expression, "hypothesis": definition["hypothesis"],
            "failed_combination_checks": row["failed_checks"], "paired_training": row.get("llm_paired_training"),
            "trade": describe_trade(trials[strategy], baseline, thresholds) if strategy in trials else None,
        })
    if not previous_expressions:
        raise ValueError("No prior weekly substitutions match the current seed controls")
    if any(file_fingerprint(path) != fingerprints[name] for name, path in paths.items()):
        raise ValueError("Weekly feedback inputs changed during read")
    return {"validation_window": window, "by_control": by_control,
            "prior_expressions": sorted(previous_expressions), "input_fingerprints": fingerprints}
