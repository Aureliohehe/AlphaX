"""Factor registry lifecycle and diagnostic signals from the active library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .factor_dsl import Candidate, candidate_from_dict, evaluate_candidate_score
from .miner_metrics import cross_section_rank

def load_registry(path: Path) -> Dict[str, object]:
    """Load the previous factor-library snapshot, if one exists."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Registry must contain a JSON object: {path}")
    return payload


def candidate_from_factor(factor: Dict[str, object]) -> Candidate:
    """Recover a safe Candidate from a persisted library entry."""
    return candidate_from_dict(
        {
            "candidate_id": factor.get("candidate_id", factor.get("id", "")),
            "op": factor.get("op", ""),
            "features": factor.get("features", []),
            "expression": factor.get("expression_spec", factor.get("expression")),
            "hypothesis": factor.get("hypothesis", ""),
        }
    )


def needs_revalidation(
    factor: Dict[str, object], as_of: pd.Timestamp, interval_days: int
) -> bool:
    stamp = factor.get("last_validated_at")
    if not stamp:
        return True
    try:
        last_validated = pd.Timestamp(str(stamp)).normalize()
    except (TypeError, ValueError):
        return True
    return (as_of.normalize() - last_validated).days >= interval_days


def library_factor(
    factor: Dict[str, object],
    previous: Optional[Dict[str, object]],
    as_of: pd.Timestamp,
    interval_days: int,
) -> Dict[str, object]:
    """Add lifecycle metadata and append an auditable validation snapshot."""
    payload = dict(factor)
    payload["created_at"] = (
        previous.get("created_at", as_of.strftime("%Y-%m-%d")) if previous else as_of.strftime("%Y-%m-%d")
    )
    payload["last_validated_at"] = as_of.strftime("%Y-%m-%d")
    previous_count = 0
    if previous:
        try:
            previous_count = int(previous.get("validation_count", 0))
        except (TypeError, ValueError):
            previous_count = 0
    payload["validation_count"] = previous_count + 1 if previous else 1
    history = []
    if previous and isinstance(previous.get("validation_history"), list):
        history = list(previous["validation_history"])
    history.append(
        {
            "validated_at": as_of.strftime("%Y-%m-%d"),
            "accepted": True,
            "direction": payload.get("direction"),
            "horizon_metrics": payload.get("horizon_metrics", {}),
        }
    )
    payload["validation_history"] = history
    payload["last_validation"] = history[-1]
    payload["status"] = "active"
    payload["next_validation_at"] = (
        as_of + pd.Timedelta(days=interval_days)
    ).strftime("%Y-%m-%d")
    return payload


def build_accepted_signal(
    frame: pd.DataFrame, accepted: Sequence[Dict[str, object]]
) -> pd.DataFrame:
    """Build an equal-weight composite from validation-approved factors."""
    if not accepted:
        # A maintenance round may legitimately prune the entire library. Keep
        # an auditable empty signal instead of losing the registry snapshot.
        return pd.DataFrame(columns=["datetime", "instrument", "prediction"])
    scores = []
    for factor in accepted:
        # Rehydrate the complete candidate, including expression_spec.  AST
        # factors used to be downgraded to op='expression' here, which made
        # accepted temporal factors fail while writing predictions.csv.
        candidate = candidate_from_factor(factor)
        direction = int(factor["direction"])
        scores.append(
            direction * cross_section_rank(evaluate_candidate_score(candidate, frame), frame["date"])
        )
    composite = (pd.concat(scores, axis=1).sum(axis=1, min_count=len(scores)) / len(scores)).rename("prediction")
    signal = pd.DataFrame(
        {
            "datetime": frame["date"].to_numpy(),
            "instrument": frame["ticker"].to_numpy(),
            "prediction": composite.to_numpy(dtype=float),
        }
    )
    return signal.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def maintain_library(
    candidates: Sequence[Candidate],
    previous_factors: Sequence[dict],
    current_accepted: Sequence[dict],
    results: Sequence[dict],
    as_of: pd.Timestamp,
    interval_days: int,
    *,
    generated_candidate_count: int,
    invalid_previous_factors: list[str],
) -> tuple[list[dict], dict]:
    """Carry, revalidate, or retire factors and summarize this round."""
    previous_by_id = {
        str(factor["candidate_id"]): factor
        for factor in previous_factors
        if isinstance(factor, dict) and factor.get("candidate_id")
    }
    current_accepted_by_id = {
        str(factor["candidate_id"]): factor for factor in current_accepted
    }
    active_factors: List[Dict[str, object]] = []
    retained_without_revalidation: List[str] = []
    revalidated_factors: List[str] = []
    pruned_factors: List[str] = []
    pruned_details: List[Dict[str, object]] = []
    result_by_id = {str(result["candidate_id"]): result for result in results}
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        previous = previous_by_id.get(candidate_id)
        if previous and not needs_revalidation(previous, as_of, interval_days):
            carried = dict(previous)
            carried["status"] = "active"
            carried["revalidation_due"] = False
            if not carried.get("next_validation_at"):
                last_validated = pd.Timestamp(str(carried["last_validated_at"]))
                carried["next_validation_at"] = (
                    last_validated + pd.Timedelta(days=interval_days)
                ).strftime("%Y-%m-%d")
            active_factors.append(carried)
            retained_without_revalidation.append(candidate_id)
        elif candidate_id in current_accepted_by_id:
            active_factors.append(
                library_factor(
                    current_accepted_by_id[candidate_id],
                    previous,
                    as_of,
                    interval_days,
                )
            )
            if previous:
                revalidated_factors.append(candidate_id)
        elif previous:
            pruned_factors.append(candidate_id)
            retired = dict(previous)
            retired["status"] = "retired"
            retired["retired_at"] = as_of.strftime("%Y-%m-%d")
            retired["retirement_reason"] = "failed_revalidation_gate"
            current_result = result_by_id.get(candidate_id)
            if current_result is not None:
                retired["last_failed_metrics"] = current_result.get("horizons", {})
            pruned_details.append(retired)

    active_factors.sort(
        key=lambda item: float(item.get("validation_directional_rank_ic", float("-inf"))),
        reverse=True,
    )
    maintenance = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "interval_days": interval_days,
        "generated_candidate_count": generated_candidate_count,
        "candidate_count_including_library": len(candidates),
        "previous_active_count": len(previous_factors),
        "current_round_gate_pass_count": len(current_accepted),
        "new_or_revalidated_count": len(
            [factor for factor in current_accepted if str(factor["candidate_id"]) not in previous_by_id]
        ) + len(revalidated_factors),
        "active_count": len(active_factors),
        "retained_without_revalidation": retained_without_revalidation,
        "revalidated": revalidated_factors,
        "pruned": pruned_factors,
        "pruned_details": pruned_details,
        "invalid_previous_factors": invalid_previous_factors,
    }
    return active_factors, maintenance
