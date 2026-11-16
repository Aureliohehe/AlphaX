"""Budgeted proposal feedback using training data only; never grants admission."""
from dataclasses import asdict
import json

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
    ranked = frame[available].groupby(dates, sort=False).rank(pct=True)
    score = evaluate_candidate_score(candidate, ranked, frame, dates)
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


def run_search(propose, frame, available, targets, label_ends, rounds=3, budget=40, batch_size=10, folds=3):
    """propose(count, feedback) returns safe candidates; every proposal consumes budget."""
    if min(rounds, budget, batch_size) < 1:
        raise ValueError("Search rounds, budget and batch size must be positive")
    if folds < 2 or frame["date"].nunique() < (folds + 1) * 5:
        raise ValueError("Search needs >=2 folds and >=5 sessions per block")
    accepted, feedback, audit, seen, ids = [], [], [], set(), set()
    spent = 0
    for round_index in range(rounds):
        count = min(batch_size, budget - spent)
        if count <= 0:
            break
        proposals = list(propose(count, json.dumps(feedback, ensure_ascii=False, allow_nan=False)))
        # Reserve the requested count even when providers return fewer or duplicates.
        spent += count
        if len(proposals) > count:
            proposals = proposals[:count]
        current = []
        for candidate in proposals:
            if candidate.expression in seen or candidate.candidate_id in ids:
                continue
            seen.add(candidate.expression)
            ids.add(candidate.candidate_id)
            accepted.append(candidate)
            report = rolling_feedback(frame, candidate, available, targets, label_ends, folds)
            # JSON nulls instead of NaN for absent diagnostic metrics.
            for fold in report["folds"]:
                fold["metrics"] = {k: v if np.isfinite(v) else None for k, v in fold["metrics"].items()}
            report["round"] = round_index + 1
            current.append(report)
        feedback.extend(current)
        audit.append({"round": round_index + 1, "requested": count, "returned": len(proposals),
                      "unique": len(current), "budget_spent": spent})
    return accepted, {"scope": "training_only_research_not_admission", "budget": budget,
                      "budget_spent": spent, "rounds": audit, "candidates": feedback}
