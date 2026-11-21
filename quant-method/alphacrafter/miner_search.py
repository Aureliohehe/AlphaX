"""Budgeted proposal feedback using training data only; never grants admission."""
from dataclasses import asdict
import json
import re

import numpy as np
import pandas as pd


def rolling_feedback(frame, candidate, available, targets, label_ends, folds=3):
    # Import lazily so the CLI can reuse the same evaluator without a cycle.
    from miner_lite import evaluate_candidate_score, evaluate_score, PAPER_HORIZON_THRESHOLDS

    dates = frame["date"]
    sessions = np.sort(dates.unique())
    if folds < 2 or len(sessions) < (folds + 1) * 5:
        raise ValueError("Search needs >=2 folds and >=5 sessions per block")
    blocks = np.array_split(sessions, folds + 1)
    score = evaluate_candidate_score(candidate, frame, dates)
    records = []
    directions = []
    for index, block in enumerate(blocks[1:], 1):
        prefix_end = pd.Timestamp(blocks[index - 1][-1])
        prefix = (dates <= prefix_end) & label_ends[1].notna() & (label_ends[1] <= prefix_end)
        fit = evaluate_score(dates, score, targets[1], prefix, frame["ticker"])
        direction = (1 if fit.mean_daily_rank_ic >= 0 else -1) if np.isfinite(fit.mean_daily_rank_ic) else None
        directions.append(direction)
        for horizon, target in targets.items():
            end = pd.Timestamp(block[-1])
            mask = dates.between(block[0], end) & label_ends[horizon].notna() & (label_ends[horizon] <= end)
            metric = asdict(evaluate_score(dates, score, target, mask, frame["ticker"]))
            reasons = []
            thresholds = PAPER_HORIZON_THRESHOLDS[horizon]
            for key, threshold in (("mean_daily_rank_ic", thresholds["rank_ic"]), ("rank_icir", thresholds["rank_icir"])):
                if direction is None or not np.isfinite(metric[key]) or direction * metric[key] <= threshold:
                    reasons.append("insufficient_" + key)
            hit = metric["rank_ic_hit_ratio"]
            if direction is None or not np.isfinite(hit) or (hit if direction > 0 else 1 - hit) <= .60:
                reasons.append("unstable_daily_direction")
            if not np.isfinite(metric["coverage"]) or metric["coverage"] <= .90:
                reasons.append("low_coverage")
            if not np.isfinite(metric["turnover"]) or metric["turnover"] >= .40:
                reasons.append("high_or_unknown_turnover")
            records.append({"fold": index, "horizon": horizon, "start": str(pd.Timestamp(block[0]).date()),
                            "end": str(end.date()), "direction_fit_end": str(prefix_end.date()),
                            "direction": direction, "metrics": metric, "failure_reasons": reasons})
    return {"candidate_id": candidate.candidate_id, "expression": candidate.expression,
            "hypothesis": candidate.hypothesis, "direction_changes": sum(a != b for a, b in zip(directions, directions[1:])),
            "folds": records}


def validation_tasks(report):
    """Turn search-fold failures into controlled follow-up experiments."""
    tasks = []
    for horizon in sorted({fold["horizon"] for fold in report["folds"]}):
        folds = [fold for fold in report["folds"] if fold["horizon"] == horizon]
        failures = [fold for fold in folds if fold["failure_reasons"]]
        reasons = {reason for fold in failures for reason in fold["failure_reasons"]}
        if "low_coverage" in reasons:
            action = "Simplify one input or operator to improve coverage"
            metric = "coverage"
        elif report["direction_changes"] or "unstable_daily_direction" in reasons:
            action = "Test one longer lookback or smoothing window for directional stability"
            metric = "rank_ic_hit_ratio"
        elif "high_or_unknown_turnover" in reasons:
            action = "Test one smoothing change to reduce rank turnover"
            metric = "turnover"
        elif failures:
            action = "Test one economically motivated input substitution for stronger rank IC"
            metric = "mean_daily_rank_ic"
        else:
            action = "Keep as a control for a distinct hypothesis"
            metric = None
        tasks.append({
            "control_id": report["candidate_id"], "horizon": horizon,
            "failing_folds": [fold["fold"] for fold in failures],
            "action": action, "primary_metric": metric,
            "comparison": "Compare to the unchanged control on the same folds with the existing thresholds",
        })
    return tasks


def changed_fields(control, variant):
    """Count scalar edits, or return None when the AST shape changes."""
    if type(control) is not type(variant):
        return None
    if isinstance(control, dict):
        if control.keys() != variant.keys():
            return None
        changes = [changed_fields(control[key], variant[key]) for key in control]
    elif isinstance(control, list):
        if len(control) != len(variant):
            return None
        changes = [changed_fields(left, right) for left, right in zip(control, variant)]
    else:
        return int(control != variant)
    return None if None in changes else sum(changes)


def feature_family(candidate):
    """Treat different lookbacks of the same inputs as one research family."""
    return frozenset(re.sub(r"\d+", "", feature) for feature in candidate.features)


def stable_horizons(report):
    """Require a majority of folds to favor the same fitted direction."""
    stable = []
    for horizon in sorted({fold["horizon"] for fold in report["folds"]}):
        folds = [fold for fold in report["folds"] if fold["horizon"] == horizon]
        support = {-1: 0, 1: 0}
        for fold in folds:
            direction = fold["direction"]
            rank_ic = fold["metrics"]["mean_daily_rank_ic"]
            if direction is not None and rank_ic is not None and direction * rank_ic > 0:
                support[direction] += 1
        if max(support.values()) > len(folds) / 2:
            stable.append(horizon)
    return stable


def fresh_feature_controls(available, families, ids, count):
    """Spend replacement slots on distinct, previously unused input families."""
    from miner_lite import Candidate

    used_inputs = set().union(*families)
    controls = []
    for index, feature in enumerate(sorted(available)):
        cid = f"reserve_feature_{index}"
        control = Candidate(cid, "feature", (feature,),
                            f"Test the standalone cross-sectional signal from {feature}.")
        family = feature_family(control)
        if cid in ids or not family.isdisjoint(used_inputs):
            continue
        controls.append(control)
        used_inputs.update(family)
        if len(controls) == count:
            break
    return controls


def run_search(propose, frame, available, targets, label_ends, rounds=3, budget=40, batch_size=10, folds=3, reserve=0):
    """propose(count, feedback) returns safe candidates; every proposal consumes budget."""
    if reserve < 0 or reserve >= budget:
        raise ValueError("Search reserve must be nonnegative and smaller than the total budget")
    if min(rounds, budget, batch_size) < 1:
        raise ValueError("Search rounds, budget and batch size must be positive")
    if folds < 2 or frame["date"].nunique() < (folds + 1) * 5:
        raise ValueError("Search needs >=2 folds and >=5 sessions per block")
    accepted, feedback, audit, seen, ids = [], [], [], set(), set()
    from miner_lite import _candidate_expression_spec

    history, families, used_lineages = {}, {}, set()
    spent, missing = 0, 0
    stable_controls = set()
    regular_rounds = min(rounds, (budget - reserve + batch_size - 1) // batch_size)
    for round_index in range(regular_rounds + reserve):
        supplemental = round_index >= regular_rounds
        if supplemental and not missing:
            break
        count = min(batch_size, budget - spent, missing) if supplemental else min(batch_size, budget - reserve - spent)
        if count <= 0:
            break
        feedback_ids = [item["candidate_id"] for item in feedback] if not supplemental else []
        followup_limit = count // 2 if stable_controls and not supplemental else 0
        context = feedback
        if reserve:
            context = [*feedback, {"round_policy": {"max_followups": followup_limit,
                                                     "min_new_hypotheses": count - followup_limit},
                                   "previous_rejections": audit[-1]["rejected"] if audit else []}]
        if supplemental:
            proposals = fresh_feature_controls(available, families, ids, count)
        else:
            proposals = list(propose(count, json.dumps(context, ensure_ascii=False, allow_nan=False)))
        # Reserve the requested count even when providers return fewer or duplicates.
        spent += count
        if len(proposals) > count:
            proposals = proposals[:count]
        current, rejected = [], []
        followups = 0
        prior = history.copy()
        for candidate in proposals:
            cid, parent = candidate.candidate_id, candidate.control_id
            spec = _candidate_expression_spec(candidate)
            family = feature_family(candidate)
            edits = None
            lineage = cid
            reason = None
            if candidate.expression in seen or cid in ids:
                reason = "duplicate_id_or_expression"
            elif parent is not None:
                if parent not in prior:
                    reason = "control_must_be_from_an_earlier_round"
                else:
                    control, lineage = prior[parent]
                    edits = changed_fields(control, spec)
                    if edits != 1:
                        reason = "follow_up_must_change_exactly_one_ast_field"
                    elif lineage in used_lineages:
                        reason = "lineage_follow_up_limit_reached"
                    elif parent not in stable_controls:
                        reason = "control_lacks_directional_support"
                    elif followups >= followup_limit:
                        reason = "follow_up_quota_reached"
                    elif family in families and families[family] != lineage:
                        reason = "feature_family_already_in_another_lineage"
            elif family in families:
                reason = "related_hypothesis_requires_control"
            if reason:
                rejected.append({"candidate_id": cid, "control_id": parent,
                                 "reason": reason, "ast_edits": edits})
                continue
            seen.add(candidate.expression)
            ids.add(cid)
            history[cid] = (spec, lineage)
            families[family] = lineage
            if parent is not None:
                used_lineages.add(lineage)
                followups += 1
            accepted.append(candidate)
            report = rolling_feedback(frame, candidate, available, targets, label_ends, folds)
            # JSON nulls instead of NaN for absent diagnostic metrics.
            for fold in report["folds"]:
                fold["metrics"] = {k: v if np.isfinite(v) else None for k, v in fold["metrics"].items()}
            report["round"] = round_index + 1
            report["control_id"] = parent
            report["lineage_id"] = lineage
            report["expression_ast"] = spec
            report["next_validation_tasks"] = validation_tasks(report)
            report["stable_horizons"] = stable_horizons(report)
            if report["stable_horizons"]:
                stable_controls.add(cid)
            current.append(report)
        feedback.extend(current)
        for report in feedback:
            report["lineage_followup_used"] = report["lineage_id"] in used_lineages
        missing += (0 if supplemental else count) - len(current)
        audit.append({"round": round_index + 1, "requested": count, "returned": len(proposals),
                      "feedback_candidate_ids": feedback_ids,
                      "supplemental": supplemental, "followup_limit": followup_limit,
                      "unique": len(current), "rejected": rejected, "budget_spent": spent})
    return accepted, {"scope": "training_only_research_not_admission", "budget": budget,
                      "budget_spent": spent, "reserve": reserve, "rounds": audit, "candidates": feedback}
