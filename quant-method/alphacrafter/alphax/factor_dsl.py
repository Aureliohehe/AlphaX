"""Parse, validate and evaluate AlphaX's JSON factor expression DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ALLOWED_OPS = {"feature", "difference", "sum"}
EXPRESSION_OPS = {
    "feature",
    "rank",
    "ts_mean",
    "ts_std",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_quantile",
    "ts_delta",
    "ts_slope",
    "ts_rsquare",
    "ts_resi",
    "ts_zscore",
    "ts_rank",
    "ts_corr",
    "sum",
    "difference",
    "product",
    "divide",
    "negate",
    "abs",
    "log",
}
MAX_EXPRESSION_DEPTH = 4
MIN_ROLLING_WINDOW = 2
MAX_ROLLING_WINDOW = 252


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    op: str
    features: Tuple[str, ...]
    hypothesis: str
    expression_spec: Optional[Dict[str, object]] = None
    control_id: Optional[str] = None
    iteration_note: Optional[Dict[str, object]] = None

    @property
    def expression(self) -> str:
        return render_expression(candidate_expression_spec(self))


def expression_features(node: Dict[str, object]) -> Tuple[str, ...]:
    """Extract leaf features from the safe expression AST."""
    op = node.get("op")
    if op == "feature":
        return (str(node.get("feature", "")),)
    if op == "ts_corr":
        values = node.get("features", [])
        return tuple(str(value) for value in values) if isinstance(values, list) else ()
    if op in {
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_quantile",
        "ts_delta",
        "ts_slope",
        "ts_rsquare",
        "ts_resi",
        "ts_zscore",
        "ts_rank",
    }:
        return (str(node.get("feature", "")),)
    features: List[str] = []
    child = node.get("input")
    if isinstance(child, dict):
        features.extend(expression_features(child))
    children = node.get("inputs")
    if isinstance(children, list):
        for value in children:
            if isinstance(value, dict):
                features.extend(expression_features(value))
    return tuple(dict.fromkeys(features))


def validate_expression(
    node: object, available: Optional[Iterable[str]] = None, depth: int = 0
) -> None:
    if not isinstance(node, dict):
        raise ValueError("expression nodes must be JSON objects")
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError(f"expression depth cannot exceed {MAX_EXPRESSION_DEPTH}")
    if available is not None and not isinstance(available, (set, frozenset)):
        available = frozenset(available)
    op = node.get("op")
    if not isinstance(op, str) or op not in EXPRESSION_OPS:
        raise ValueError(f"Unsupported expression op: {op}")
    if op == "feature":
        feature = node.get("feature")
        if not isinstance(feature, str) or not feature:
            raise ValueError("feature expression requires a feature name")
    elif op == "ts_corr":
        features = node.get("features")
        if not isinstance(features, list) or len(features) != 2 or not all(
            isinstance(value, str) and value for value in features
        ):
            raise ValueError("ts_corr requires exactly two feature names")
    elif op in {
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_quantile",
        "ts_delta",
        "ts_slope",
        "ts_rsquare",
        "ts_resi",
        "ts_zscore",
        "ts_rank",
    }:
        feature = node.get("feature")
        if not isinstance(feature, str) or not feature:
            raise ValueError(f"{op} requires one feature name")
    elif op in {"rank", "negate", "abs", "log"}:
        child = node.get("input")
        if not isinstance(child, dict):
            raise ValueError(f"{op} requires an input expression")
        validate_expression(child, available, depth + 1)
    else:
        children = node.get("inputs")
        if not isinstance(children, list) or len(children) != 2 or not all(
            isinstance(value, dict) for value in children
        ):
            raise ValueError(f"{op} requires exactly two input expressions")
        for child in children:
            validate_expression(child, available, depth + 1)

    if op.startswith("ts_"):
        window = node.get("window")
        if not isinstance(window, int) or isinstance(window, bool):
            raise ValueError(f"{op} requires an integer window")
        if not MIN_ROLLING_WINDOW <= window <= MAX_ROLLING_WINDOW:
            raise ValueError(
                f"rolling window must be between {MIN_ROLLING_WINDOW} and {MAX_ROLLING_WINDOW}"
            )
        if op == "ts_quantile":
            quantile = node.get("quantile")
            if not isinstance(quantile, (int, float)) or not 0 <= quantile <= 1:
                raise ValueError("ts_quantile requires quantile between 0 and 1")
    if available is not None:
        missing = sorted(set(expression_features(node)).difference(available))
        if missing:
            raise ValueError(f"Expression uses unavailable features: {missing}")


def candidate_expression_spec(candidate: Candidate) -> Dict[str, object]:
    """Canonical AST for evaluation, rendering, persistence and novelty analysis."""
    if candidate.expression_spec is not None:
        return candidate.expression_spec
    if candidate.op == "feature":
        return {"op": "feature", "feature": candidate.features[0]}
    return {
        "op": candidate.op,
        "inputs": [
            {"op": "feature", "feature": candidate.features[0]},
            {"op": "feature", "feature": candidate.features[1]},
        ],
    }


def render_expression(node: Dict[str, object]) -> str:
    op = str(node["op"])
    if op == "feature":
        return str(node["feature"])
    if op in {"rank", "negate", "abs", "log"}:
        rendered = render_expression(node["input"])
        if op == "rank":
            return f"rank({rendered})"
        if op == "negate":
            return f"-({rendered})"
        return f"{op}({rendered})"
    if op in {
        "ts_mean",
        "ts_std",
        "ts_sum",
        "ts_min",
        "ts_max",
        "ts_delta",
        "ts_slope",
        "ts_rsquare",
        "ts_resi",
        "ts_zscore",
        "ts_rank",
    }:
        return f"{op}({node['feature']}, {node['window']})"
    if op == "ts_quantile":
        return f"ts_quantile({node['feature']}, {node['window']}, {node['quantile']})"
    if op == "ts_corr":
        return f"ts_corr({node['features'][0]}, {node['features'][1]}, {node['window']})"
    symbols = {"sum": "+", "difference": "-", "product": "*", "divide": "/"}
    left, right = node["inputs"]
    return f"({render_expression(left)} {symbols[op]} {render_expression(right)})"


def normalize_expression(node: Dict[str, object]) -> Dict[str, object]:
    """Normalize common equivalent AST spellings before strict validation."""
    return _normalize_expression(node, depth=0)


def _normalize_expression(node: Dict[str, object], depth: int) -> Dict[str, object]:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ValueError(f"expression depth cannot exceed {MAX_EXPRESSION_DEPTH}")
    normalized = dict(node)
    op = normalized.get("op")
    if op in ("rank", "negate") and "input" not in normalized:
        feature = normalized.pop("feature", None)
        if isinstance(feature, str):
            normalized["input"] = {"op": "feature", "feature": feature}
    if op in ("sum", "difference", "product", "divide") and "inputs" not in normalized:
        left = normalized.pop("left", None)
        right = normalized.pop("right", None)
        if isinstance(left, dict) and isinstance(right, dict):
            normalized["inputs"] = [left, right]
        else:
            features = normalized.get("features")
            if isinstance(features, list) and len(features) == 2 and all(
                isinstance(feature, str) for feature in features
            ):
                normalized["inputs"] = [
                    {"op": "feature", "feature": features[0]},
                    {"op": "feature", "feature": features[1]},
                ]
    if op == "ts_corr":
        features = normalized.get("features")
        if isinstance(features, list) and all(isinstance(feature, dict) for feature in features):
            normalized["features"] = [feature.get("feature") for feature in features]
    if isinstance(normalized.get("input"), dict):
        normalized["input"] = _normalize_expression(normalized["input"], depth + 1)
    if isinstance(normalized.get("inputs"), list):
        normalized["inputs"] = [
            _normalize_expression(child, depth + 1) if isinstance(child, dict) else child
            for child in normalized["inputs"]
        ]
    return normalized


def parse_expression(
    value: object, available: Optional[Iterable[str]] = None
) -> Dict[str, object]:
    """Normalize and validate an expression decoded from JSON."""
    if not isinstance(value, dict):
        raise ValueError("expression nodes must be JSON objects")
    expression = normalize_expression(value)
    validate_expression(expression, available)
    return expression


def candidate_from_dict(value: Dict[str, object]) -> Candidate:
    """Validate both the legacy candidate schema and the safe AST schema."""
    expression_value = value.get("expression")
    expression_spec = normalize_expression(expression_value) if isinstance(expression_value, dict) else None
    op = "expression" if expression_spec is not None else str(value.get("op", ""))
    features_value = value.get("features")
    if features_value is None and expression_spec is not None:
        features_value = list(expression_features(expression_spec))
    if not isinstance(features_value, list) or not all(isinstance(x, str) for x in features_value):
        raise ValueError("candidate.features must be a list of strings")
    control_id = value.get("control_id")
    if control_id is not None and (not isinstance(control_id, str) or not control_id.strip()):
        raise ValueError("control_id must be a nonempty string or null")
    candidate = Candidate(
        candidate_id=str(value.get("candidate_id", value.get("id", ""))),
        op=op,
        features=tuple(features_value),
        hypothesis=str(value.get("hypothesis", "")),
        expression_spec=expression_spec,
        control_id=control_id,
        iteration_note=value.get("iteration_note") if isinstance(value.get("iteration_note"), dict) else None,
    )
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: Candidate, available: Optional[Iterable[str]] = None) -> None:
    if not candidate.candidate_id:
        raise ValueError("candidate_id cannot be empty")
    if candidate.expression_spec is not None:
        validate_expression(candidate.expression_spec, available)
        expression_leafs = set(expression_features(candidate.expression_spec))
        if set(candidate.features) != expression_leafs:
            raise ValueError("candidate.features must match expression leaf features")
        return
    if candidate.op not in ALLOWED_OPS:
        raise ValueError(f"Unsupported op: {candidate.op}")
    expected = 1 if candidate.op == "feature" else 2
    if len(candidate.features) != expected:
        raise ValueError(f"{candidate.op} expects {expected} features")
    if available is not None:
        missing = sorted(set(candidate.features) - set(available))
        if missing:
            raise ValueError(f"Candidate uses unavailable features: {missing}")


def _rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = np.dot(centered, centered)
    return float(np.dot(centered, values - values.mean()) / denominator) if denominator else np.nan


def _rolling_rsquare(values: np.ndarray) -> float:
    slope = _rolling_slope(values)
    x = np.arange(len(values), dtype=float)
    intercept = values.mean() - slope * x.mean()
    total = np.sum((values - values.mean()) ** 2)
    return float(1.0 - np.sum((values - (intercept + slope * x)) ** 2) / total) if total else 0.0


def _rolling_residual(values: np.ndarray) -> float:
    slope = _rolling_slope(values)
    x = np.arange(len(values), dtype=float)
    intercept = values.mean() - slope * x.mean()
    return float(values[-1] - (intercept + slope * x[-1]))


def _time_series_values(
    frame: pd.DataFrame,
    values: pd.Series,
    operation: str,
    window: int,
    quantile: Optional[float] = None,
) -> pd.Series:
    ordered = pd.DataFrame(
        {"ticker": frame["ticker"], "date": frame["date"], "value": values},
        index=frame.index,
    ).sort_values(["ticker", "date"])
    grouped = ordered.groupby("ticker", sort=False)["value"]
    if operation == "ts_mean":
        transformed = grouped.transform(lambda series: series.rolling(window, min_periods=window).mean())
    elif operation == "ts_std":
        transformed = grouped.transform(lambda series: series.rolling(window, min_periods=window).std())
    elif operation == "ts_sum":
        transformed = grouped.transform(lambda series: series.rolling(window, min_periods=window).sum())
    elif operation == "ts_min":
        transformed = grouped.transform(lambda series: series.rolling(window, min_periods=window).min())
    elif operation == "ts_max":
        transformed = grouped.transform(lambda series: series.rolling(window, min_periods=window).max())
    elif operation == "ts_quantile":
        transformed = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).quantile(float(quantile))
        )
    elif operation == "ts_delta":
        transformed = grouped.transform(lambda series: series - series.shift(window))
    elif operation == "ts_slope":
        transformed = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).apply(_rolling_slope, raw=True)
        )
    elif operation == "ts_rsquare":
        transformed = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).apply(_rolling_rsquare, raw=True)
        )
    elif operation == "ts_resi":
        transformed = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).apply(_rolling_residual, raw=True)
        )
    elif operation == "ts_zscore":
        mean = grouped.transform(lambda series: series.rolling(window, min_periods=window).mean())
        std = grouped.transform(lambda series: series.rolling(window, min_periods=window).std())
        transformed = (ordered["value"] - mean) / std.replace(0, np.nan)
    elif operation == "ts_rank":
        transformed = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).apply(
                lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=True
            )
        )
    else:
        raise ValueError(f"Unsupported time-series operation: {operation}")
    return pd.Series(transformed.to_numpy(), index=ordered.index).reindex(frame.index)


def _time_series_corr(
    frame: pd.DataFrame, left: pd.Series, right: pd.Series, window: int
) -> pd.Series:
    ordered = pd.DataFrame(
        {
            "ticker": frame["ticker"],
            "date": frame["date"],
            "left": left,
            "right": right,
        },
        index=frame.index,
    ).sort_values(["ticker", "date"])
    output = pd.Series(np.nan, index=ordered.index, dtype=float)
    for _, indices in ordered.groupby("ticker", sort=False).groups.items():
        group = ordered.loc[indices]
        output.loc[indices] = group["left"].rolling(window, min_periods=window).corr(group["right"])
    return output.reindex(frame.index)


def _evaluate_expression(
    node: Dict[str, object], frame: pd.DataFrame, dates: pd.Series
) -> pd.Series:
    op = str(node["op"])
    if op == "feature":
        return frame[str(node["feature"])]
    if op == "rank":
        child = _evaluate_expression(node["input"], frame, dates)
        return child.groupby(dates, sort=False).rank(pct=True)
    if op.startswith("ts_") and op != "ts_corr":
        return _time_series_values(
            frame,
            frame[str(node["feature"])],
            op,
            int(node["window"]),
            float(node["quantile"]) if op == "ts_quantile" else None,
        )
    if op == "ts_corr":
        left, right = node["features"]
        return _time_series_corr(
            frame,
            frame[str(left)],
            frame[str(right)],
            int(node["window"]),
        )
    if op == "negate":
        return -_evaluate_expression(node["input"], frame, dates)
    if op == "abs":
        return _evaluate_expression(node["input"], frame, dates).abs()
    if op == "log":
        return np.log1p(_evaluate_expression(node["input"], frame, dates).clip(lower=-0.999999))
    left = _evaluate_expression(node["inputs"][0], frame, dates)
    right = _evaluate_expression(node["inputs"][1], frame, dates)
    if op == "sum":
        return left + right
    if op == "difference":
        return left - right
    if op == "product":
        return left * right
    if op == "divide":
        return left / right.replace(0, np.nan)
    raise ValueError(f"Unsupported expression operation: {op}")


def evaluate_candidate_score(
    candidate: Candidate,
    frame: pd.DataFrame,
    dates: Optional[pd.Series] = None,
) -> pd.Series:
    """Evaluate the declared expression on raw fields; rank only when explicit."""
    validate_candidate(candidate, frame.columns)
    return _evaluate_expression(
        candidate_expression_spec(candidate), frame,
        frame["date"] if dates is None else dates,
    )


def evaluate_expression(
    value: object, frame: pd.DataFrame, dates: Optional[pd.Series] = None
) -> pd.Series:
    """Evaluate a JSON expression on raw panel fields, aligned to the input index."""
    expression = parse_expression(value, frame.columns)
    return _evaluate_expression(expression, frame, frame["date"] if dates is None else dates)
