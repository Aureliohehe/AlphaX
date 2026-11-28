"""Candidate templates, input validation, and expression novelty diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .factor_dsl import (
    Candidate,
    candidate_expression_spec,
    candidate_from_dict,
    expression_features,
    parse_expression,
    validate_candidate,
)

BASELINE_FEATURES = {"return_1d", "one_day_reversal"}


# Related features are paired only within a family.  This keeps the first
# search interpretable and prevents a combinatorial explosion.
FEATURE_FAMILIES = {
    "momentum": [
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "sma_5_ratio",
        "sma_10_ratio",
        "sma_20_ratio",
        "sma_60_ratio",
        "ema_12_ratio",
        "ema_26_ratio",
    ],
    "range_reversal": [
        "open_close_return",
        "close_location",
        "stochastic_14",
        "bollinger_z",
        "drawdown_from_high_5d",
        "drawdown_from_high_20d",
        "distance_from_low_5d",
        "distance_from_low_20d",
    ],
    "volatility": [
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "atr_14_ratio",
        "bollinger_width",
    ],
    "volume": [
        "volume_ratio_20d",
        "volume_change_1d",
        "volume_volatility_20d",
    ],
}


def _expression_tokens(node: Dict[str, object]) -> List[str]:
    """Pre-order AST topology tokens used by the novelty diagnostic."""
    op = str(node.get("op"))
    if op == "feature":
        return ["feature"]
    if op == "ts_corr":
        return [op, "feature", "feature"]
    if op.startswith("ts_"):
        return [op, "feature"]
    tokens = [op]
    child = node.get("input")
    if isinstance(child, dict):
        tokens.extend(_expression_tokens(child))
    children = node.get("inputs")
    if isinstance(children, list):
        for value in children:
            if isinstance(value, dict):
                tokens.extend(_expression_tokens(value))
    return tokens


def _sequence_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def expression_novelty_report(
    candidates: Sequence[Candidate], baseline: Sequence[Candidate] = ()
) -> Dict[str, object]:
    """Report normalized structural diversity, following the paper's case study."""
    tokenized = {
        candidate.candidate_id: _expression_tokens(candidate_expression_spec(candidate))
        for candidate in candidates
    }

    def distance(left: Sequence[str], right: Sequence[str]) -> float:
        denominator = max(1, len(left) + len(right))
        return float(_sequence_edit_distance(left, right) / denominator)

    pair_distances = [
        distance(tokenized[left.candidate_id], tokenized[right.candidate_id])
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
    ]
    baseline_tokens = [
        _expression_tokens(candidate_expression_spec(candidate)) for candidate in baseline
    ]
    nearest_baseline = []
    if baseline_tokens:
        for candidate in candidates:
            nearest_baseline.append(
                min(distance(tokenized[candidate.candidate_id], tokens) for tokens in baseline_tokens)
            )
    return {
        "candidate_count": len(candidates),
        "baseline_count": len(baseline),
        "internal_structural_diversity": (
            float(np.mean(pair_distances)) if pair_distances else np.nan
        ),
        "external_structural_novelty": (
            float(np.mean(nearest_baseline)) if nearest_baseline else np.nan
        ),
        "max_external_structural_novelty": (
            float(np.max(nearest_baseline)) if nearest_baseline else np.nan
        ),
    }


def temporal_candidates(available: Sequence[str], max_count: int) -> List[Candidate]:
    """Generate deterministic candidates using safe time-series AST operators."""
    available_set = set(available)
    specs = [
        ("ts_zscore", "return_5d", 20),
        ("ts_zscore", "return_20d", 20),
        ("ts_delta", "return_5d", 5),
        ("ts_delta", "return_20d", 5),
        ("ts_mean", "volatility_20d", 20),
        ("ts_std", "return_5d", 20),
        ("ts_sum", "return_5d", 20),
        ("ts_min", "return_5d", 20),
        ("ts_max", "return_5d", 20),
        ("ts_slope", "return_20d", 20),
        ("ts_rsquare", "return_20d", 20),
        ("ts_resi", "return_20d", 20),
        ("ts_rank", "return_5d", 20),
    ]
    candidates: List[Candidate] = []
    for op, feature, window in specs:
        if feature not in available_set:
            continue
        spec = {"op": op, "feature": feature, "window": window}
        candidates.append(
            Candidate(
                candidate_id=f"{op}__{feature}__{window}",
                op="expression",
                features=(feature,),
                hypothesis=f"The {op} of {feature} over {window} sessions captures temporal structure.",
                expression_spec=spec,
            )
        )
    if "return_5d" in available_set and "volume_ratio_20d" in available_set:
        spec = {
            "op": "ts_corr",
            "features": ["return_5d", "volume_ratio_20d"],
            "window": 20,
        }
        candidates.append(
            Candidate(
                candidate_id="ts_corr__return_5d__volume_ratio_20d__20",
                op="expression",
                features=("return_5d", "volume_ratio_20d"),
                hypothesis="Return and volume co-movement contains temporal cross-sectional information.",
                expression_spec=spec,
            )
        )
    return candidates[:max_count]


def template_candidates(
    available: Sequence[str], max_pairs: int, max_temporal: int = 0
) -> List[Candidate]:
    """Generate an interpretable, deterministic candidate set."""
    available_set = set(available)
    candidates: List[Candidate] = []
    for feature in available:
        candidates.append(
            Candidate(
                candidate_id=f"single__{feature}",
                op="feature",
                features=(feature,),
                hypothesis=f"Cross-sectional rank of {feature} has incremental predictive information.",
            )
        )

    pairs: List[Candidate] = []
    for family, family_features in FEATURE_FAMILIES.items():
        usable = [feature for feature in family_features if feature in available_set]
        for left_index, left in enumerate(usable):
            for right in usable[left_index + 1 :]:
                pairs.append(
                    Candidate(
                        candidate_id=f"spread__{left}__{right}",
                        op="difference",
                        features=(left, right),
                        hypothesis=(
                            f"The relative {family} signal {left} versus {right} "
                            "contains more information than either level alone."
                        ),
                    )
                )

    # Stable ordering makes experiments reproducible.  The limit is a safety
    # valve for future schemas with many more fields.
    candidates.extend(sorted(pairs, key=lambda item: item.candidate_id)[:max_pairs])
    if max_temporal:
        candidates.extend(temporal_candidates(available, max_temporal))
    return candidates


def load_candidates(path: Path, available: Sequence[str]) -> List[Candidate]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("candidates", [])
    if not isinstance(payload, list):
        raise ValueError("Candidate file must contain a list or {'candidates': [...]}")
    candidates = [candidate_from_dict(value) for value in payload]
    for candidate in candidates:
        validate_candidate(candidate, available)
    return candidates


def parse_llm_candidates(payload: object, available: Sequence[str], audit: List[Dict[str, object]]) -> List[Candidate]:
    """Reject unsafe proposals individually; correct only redundant AST leaf metadata."""
    if not isinstance(payload, list):
        raise ValueError("LLM response must be a candidate array")
    accepted = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            audit.append({"index": index, "status": "rejected", "reason": "candidate is not an object"})
            continue
        candidate_id = str(value.get("candidate_id", value.get("id", "")))
        proposed = dict(value)
        if isinstance(value.get("expression"), dict):
            try:
                ast = parse_expression(value["expression"], available)
                leaves = list(expression_features(ast))
                if proposed.get("features") != leaves:
                    proposed["features"] = leaves
                    audit.append({"index": index, "candidate_id": candidate_id, "status": "normalized_features",
                                  "reason": "redundant features list differs from validated AST leaves"})
            except (ValueError, TypeError) as exc:
                audit.append({"index": index, "candidate_id": candidate_id, "status": "rejected", "reason": str(exc)})
                continue
        try:
            candidate = candidate_from_dict(proposed)
            validate_candidate(candidate, available)
        except (ValueError, TypeError) as exc:
            audit.append({"index": index, "candidate_id": candidate_id, "status": "rejected", "reason": str(exc)})
            continue
        accepted.append(candidate)
    if not accepted:
        raise ValueError("LLM proposed no valid safe candidates")
    return accepted
