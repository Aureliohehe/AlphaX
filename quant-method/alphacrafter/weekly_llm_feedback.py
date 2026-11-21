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
    return {"status": trial["status"], "cost_scenarios": breakdown,
            "paired_execution": trial.get("paired_execution")}


def load_xgboost_feedback(path: Path, as_of, control_ids):
    """Read per-factor augmentation evidence from a completed paired replay."""
    from admission import assess_weekly_substitution

    fingerprint = file_fingerprint(path)
    replay = json.loads(path.read_text())
    window = replay["window"]
    cutoff = session_date(as_of)
    if len(window) != 2 or not session_date(window[0]) < session_date(window[1]) <= cutoff:
        raise ValueError("XGBoost evaluation must end on or before the next Miner cutoff")
    by_control = {control: [] for control in control_ids} if control_ids is not None else {}
    standalone = []
    for research in replay["rounds"].values():
        if session_date(research["miner_as_of"]) >= session_date(window[0]):
            raise ValueError("XGBoost candidates must precede their evaluation window")
        for row in research["variants"]:
            control = row["control"]
            if control is None and control_ids is None:
                records = standalone
            elif control in by_control:
                records = by_control[control]
            else:
                continue
            if row.get("status") in {"rank_duplicate_in_training", "missing_complete_samples"}:
                records.append({key: row[key] for key in ("id", "expression", "status", "equivalent_to")
                                if key in row})
                continue
            baseline = row["results"]["core"]["costs"]
            augmented = row["results"]["augmented"]["costs"]
            weight = row.get("augmentation_weight", replay.get("augmentation_weight", 1.0))
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1:
                raise ValueError("XGBoost feedback requires a valid augmentation weight")
            method = row.get("augmentation_method", replay.get("augmentation_method", "refit"))
            if method not in ("refit", "factor_rank", "residual_rank"):
                raise ValueError("XGBoost feedback requires a known augmentation method")
            metrics = ("cumulative_return", "max_drawdown", "mean_daily_turnover")
            for scenarios in (baseline, augmented):
                for result in scenarios.values():
                    if not session_date(window[0]) <= session_date(result["start"]) <= session_date(result["end"]) <= session_date(window[1]):
                        raise ValueError("XGBoost trial falls outside its declared evaluation window")
                    if (any(finite_metric(result, key) is None for key in metrics)
                            or any(finite_metric(result["calendar_year_returns"], year) is None
                                   for year in result["calendar_year_returns"])):
                        raise ValueError("XGBoost feedback requires finite paired metrics")
            checks = assess_weekly_substitution(baseline, augmented, independent=False, require_each_year=True)
            records.append({
                "candidate_id": row["id"], "expression": row["expression"],
                "augmentation_weight": weight,
                "augmentation_method": method,
                **{key: row[key] for key in ("active_signal_dates", "total_signal_dates") if key in row},
                **({"activation_rule": replay["regime"]} if "regime" in replay else {}),
                "paired_execution": checks, "execution_passes": row["execution_passes"],
                "cost_deltas": {
                    cost: {metric: augmented[cost][metric] - baseline[cost][metric]
                           for metric in metrics}
                    for cost in baseline
                },
            })
    expressions = {record["expression"] for records in by_control.values() for record in records}
    expressions.update(record["expression"] for record in standalone)
    if not expressions:
        raise ValueError("No XGBoost candidates match the current proposal mode")
    if file_fingerprint(path) != fingerprint:
        raise ValueError("XGBoost feedback changed during read")
    return {"validation_window": window, "by_control": by_control, "standalone": standalone,
            "prior_expressions": sorted(expressions), "input_fingerprint": fingerprint}


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
