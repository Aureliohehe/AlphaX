#!/usr/bin/env python3
"""AlphaCrafter Miner-lite.

This module implements the first, deterministic version of the Miner Harness:

1. Generate safe candidate factor expressions from the existing feature schema.
2. Evaluate them with cross-sectional Rank IC and portfolio diagnostics.
3. Accept candidates using a train/validation gate and keep 2024+ as holdout.

An LLM is intentionally not required.  When enabled, MinerLLM proposes
candidates through the same safe JSON boundary used by ``--candidate-file``;
the validator and registry remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASELINE_FEATURES = {"return_1d", "one_day_reversal"}
META_COLUMNS = {
    "date",
    "ticker",
    "target_1d",
    "target_5d",
    "target_open_close",
    "split",
}
EVALUATION_HORIZONS = (1, 5)
ALLOWED_OPS = {"feature", "difference", "sum"}
EXPRESSION_OPS = {
    "feature",
    "rank",
    "ts_mean",
    "ts_std",
    "ts_delta",
    "ts_zscore",
    "ts_rank",
    "ts_corr",
    "sum",
    "difference",
    "product",
    "divide",
    "negate",
}
MAX_EXPRESSION_DEPTH = 4
MIN_ROLLING_WINDOW = 2
MAX_ROLLING_WINDOW = 252

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


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    op: str
    features: Tuple[str, ...]
    hypothesis: str
    expression_spec: Optional[Dict[str, object]] = None

    @property
    def expression(self) -> str:
        if self.expression_spec is not None:
            return render_expression(self.expression_spec)
        if self.op == "feature":
            return f"rank({self.features[0]})"
        if self.op == "difference":
            return f"rank({self.features[0]}) - rank({self.features[1]})"
        if self.op == "sum":
            return f"rank({self.features[0]}) + rank({self.features[1]})"
        raise ValueError(f"Unsupported operation: {self.op}")


def expression_features(node: Dict[str, object]) -> Tuple[str, ...]:
    """Extract leaf features from the safe expression AST."""
    op = node.get("op")
    if op == "feature":
        return (str(node.get("feature", "")),)
    if op == "ts_corr":
        values = node.get("features", [])
        return tuple(str(value) for value in values) if isinstance(values, list) else ()
    if op in {"ts_mean", "ts_std", "ts_delta", "ts_zscore", "ts_rank"}:
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
    op = node.get("op")
    if op not in EXPRESSION_OPS:
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
    elif op in {"ts_mean", "ts_std", "ts_delta", "ts_zscore", "ts_rank"}:
        feature = node.get("feature")
        if not isinstance(feature, str) or not feature:
            raise ValueError(f"{op} requires one feature name")
    elif op == "rank" or op == "negate":
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
    if available is not None:
        missing = sorted(set(expression_features(node)) - set(available))
        if missing:
            raise ValueError(f"Expression uses unavailable features: {missing}")


def render_expression(node: Dict[str, object]) -> str:
    op = str(node["op"])
    if op == "feature":
        return str(node["feature"])
    if op == "rank" or op == "negate":
        rendered = render_expression(node["input"])
        return f"rank({rendered})" if op == "rank" else f"-({rendered})"
    if op in {"ts_mean", "ts_std", "ts_delta", "ts_zscore", "ts_rank"}:
        return f"{op}({node['feature']}, {node['window']})"
    if op == "ts_corr":
        return f"ts_corr({node['features'][0]}, {node['features'][1]}, {node['window']})"
    symbols = {"sum": "+", "difference": "-", "product": "*", "divide": "/"}
    left, right = node["inputs"]
    return f"({render_expression(left)} {symbols[op]} {render_expression(right)})"


@dataclass
class Metric:
    rows: int
    mean_daily_rank_ic: float
    daily_rank_ic_std: float
    rank_icir: float
    rank_ic_hit_ratio: float
    mean_top_bottom_decile_spread: float
    coverage: float
    turnover: float


def normalize_expression(node: Dict[str, object]) -> Dict[str, object]:
    """Normalize common equivalent AST spellings before strict validation."""
    normalized = dict(node)
    op = normalized.get("op")
    if op in {"rank", "negate"} and "input" not in normalized:
        feature = normalized.pop("feature", None)
        if isinstance(feature, str):
            normalized["input"] = {"op": "feature", "feature": feature}
    if op in {"sum", "difference", "product", "divide"} and "inputs" not in normalized:
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
            normalized["features"] = [feature["feature"] for feature in features]
    if isinstance(normalized.get("input"), dict):
        normalized["input"] = normalize_expression(normalized["input"])
    if isinstance(normalized.get("inputs"), list):
        normalized["inputs"] = [
            normalize_expression(child) if isinstance(child, dict) else child
            for child in normalized["inputs"]
        ]
    return normalized


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
    candidate = Candidate(
        candidate_id=str(value.get("candidate_id", value.get("id", ""))),
        op=op,
        features=tuple(features_value),
        hypothesis=str(value.get("hypothesis", "")),
        expression_spec=expression_spec,
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


def build_market_context(frame: pd.DataFrame, train_mask: pd.Series) -> str:
    """Create a compact, point-in-time context for an optional LLM proposer."""
    history = frame.loc[train_mask].sort_values("date")
    if history.empty:
        return "No historical context available."
    latest_date = history["date"].max()
    recent = history[history["date"] > latest_date - pd.Timedelta(days=60)]
    fields = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "intraday_range",
    ]
    summary = {field: float(recent[field].median()) for field in fields if field in recent}
    return json.dumps(
        {
            "as_of": str(latest_date.date()),
            "window": "last 60 calendar days available in training data",
            "cross_sectional_medians": summary,
        },
        ensure_ascii=False,
    )


def generate_llm_candidates(
    model: str, available: Sequence[str], market_context: str, count: int
) -> List[Candidate]:
    """Generate candidates with DeepSeek directly or another LiteLLM provider.

    DeepSeek is implemented with the standard library because its API is
    OpenAI-compatible and the Miner should work without installing a second
    LLM framework.  Other model names retain the existing LiteLLM path.
    """
    prompt = f"""You are the Miner component of a cross-sectional quantitative research system.
Generate exactly {count} candidate factors using only the approved feature names and operations.
Do not use future data, prices after the as_of date, arbitrary Python, or unavailable fields.
A candidate must be a JSON object with:
  candidate_id: short unique snake_case string
  hypothesis: one sentence
  features: the unique leaf features used by the expression
  expression: a safe expression object, for example:
    {{"op":"ts_zscore","feature":"return_5d","window":20}}
  Allowed expression ops: feature, rank, ts_mean, ts_std, ts_delta, ts_zscore,
    ts_rank, ts_corr, sum, difference, product, divide, negate.
  AST shapes are strict: rank/negate use `input`; sum/difference/product/divide
    use `inputs` with exactly two expression objects; ts_corr uses `features`
    with exactly two feature names; time-series unary ops use `feature` and `window`.
  Time-series windows must be integers from 2 through 252.
The validator will calculate all metrics. Return only a JSON array.
Approved features: {json.dumps(list(available), ensure_ascii=False)}
Do not emit Python, arbitrary function names, or operators outside this grammar.
Point-in-time market context: {market_context}
"""
    messages = [
        {
            "role": "system",
            "content": "Return machine-readable JSON and respect the supplied schema.",
        },
        {"role": "user", "content": prompt},
    ]
    deepseek_model = model.split("/", 1)[1] if model.startswith("deepseek/") else model
    if deepseek_model in {"deepseek-chat", "deepseek-reasoner"}:
        content = _deepseek_completion(messages, deepseek_model)
    else:
        try:
            from litellm import completion
        except ImportError as exc:
            raise SystemExit(
                "This model requires LiteLLM. Install it with: python3 -m pip install litellm"
            ) from exc
        response = completion(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("LLM response content is not text")

    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(content)
    if not isinstance(payload, list):
        payload = payload.get("candidates", []) if isinstance(payload, dict) else []
    candidates = [candidate_from_dict(value) for value in payload]
    for candidate in candidates:
        validate_candidate(candidate, available)
    return candidates


def _deepseek_completion(messages: List[Dict[str, str]], model: str) -> str:
    """Call DeepSeek's OpenAI-compatible endpoint without exposing the API key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("--llm-model deepseek-chat requires DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
    request_payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4000,
    }
    # DeepSeek Reasoner does not accept sampling controls such as temperature.
    if model == "deepseek-chat":
        request_payload["temperature"] = 0.7
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach DeepSeek API: {exc.reason}") from exc

    if "error" in payload:
        raise RuntimeError(f"DeepSeek API error: {payload['error']}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("DeepSeek response did not contain message content") from exc
    if not isinstance(content, str):
        raise ValueError("DeepSeek response content is not text")
    return content


def evaluate_score(
    dates: pd.Series,
    score: pd.Series,
    target: pd.Series,
    mask: pd.Series,
    instruments: Optional[pd.Series] = None,
) -> Metric:
    """Evaluate IC plus the paper's coverage and ranking-turnover controls."""
    assets = instruments if instruments is not None else pd.Series(dates.index, index=dates.index)
    frame = pd.DataFrame(
        {"date": dates, "score": score, "target": target, "instrument": assets},
        index=dates.index,
    ).loc[mask]
    valid_scores = frame.dropna(subset=["score"])
    daily_counts = frame.groupby("date", sort=False).size()
    valid_counts = valid_scores.groupby("date", sort=False).size()
    daily_coverage = valid_counts.reindex(daily_counts.index, fill_value=0) / daily_counts
    coverage = float(daily_coverage.mean()) if not daily_coverage.empty else np.nan

    values = frame.dropna(subset=["score", "target"])
    daily_ic: List[float] = []
    spreads: List[float] = []
    for _, group in values.groupby("date", sort=False):
        if len(group) < 3 or group["score"].nunique() < 2 or group["target"].nunique() < 2:
            continue
        # Rank the score and target inside each trading day.  This is
        # equivalent to Spearman correlation and avoids scipy as a dependency.
        score_rank = group["score"].rank(method="average").to_numpy(dtype=float)
        target_rank = group["target"].rank(method="average").to_numpy(dtype=float)
        score_rank -= score_rank.mean()
        target_rank -= target_rank.mean()
        denominator = np.sqrt(np.dot(score_rank, score_rank) * np.dot(target_rank, target_rank))
        if denominator == 0:
            continue
        daily_ic.append(float(np.dot(score_rank, target_rank) / denominator))
        count = max(1, len(group) // 10)
        ordered = group.sort_values("score")
        spreads.append(
            float(ordered["target"].iloc[-count:].mean() - ordered["target"].iloc[:count].mean())
        )

    # Use percentile ranks so turnover is normalized to [0, 1], matching the
    # paper's L1 rank-change definition and its threshold of 0.4.
    ranked = valid_scores.copy()
    ranked["cross_sectional_rank"] = ranked.groupby("date", sort=False)["score"].rank(pct=True)
    dates_sorted = sorted(ranked["date"].drop_duplicates())
    previous_dates = dict(zip(dates_sorted[1:], dates_sorted[:-1]))
    ranked["previous_date"] = ranked["date"].map(previous_dates)
    previous = ranked[["date", "instrument", "cross_sectional_rank"]].rename(
        columns={"date": "previous_date", "cross_sectional_rank": "previous_rank"}
    )
    joined = ranked.dropna(subset=["previous_date"]).merge(
        previous, on=["previous_date", "instrument"], how="inner"
    )
    if joined.empty:
        turnover = np.nan
    else:
        daily_turnover = (
            (joined["cross_sectional_rank"] - joined["previous_rank"])
            .abs()
            .groupby(joined["date"], sort=False)
            .mean()
        )
        turnover = float(daily_turnover.mean()) if not daily_turnover.empty else np.nan

    ic = pd.Series(daily_ic, dtype=float).dropna()
    spread = pd.Series(spreads, dtype=float).dropna()
    if ic.empty:
        return Metric(0, np.nan, np.nan, np.nan, np.nan, np.nan, coverage, turnover)
    std = float(ic.std())
    return Metric(
        rows=int(len(values)),
        mean_daily_rank_ic=float(ic.mean()),
        daily_rank_ic_std=std,
        rank_icir=float(ic.mean() / std) if std else np.nan,
        rank_ic_hit_ratio=float((ic > 0).mean()),
        mean_top_bottom_decile_spread=float(spread.mean()),
        coverage=coverage,
        turnover=turnover,
    )


def derive_target_5d(frame: pd.DataFrame) -> pd.Series:
    """Derive a five-session forward return for legacy feature files."""
    if "target_5d" in frame:
        return frame["target_5d"]
    ordered = frame.sort_values(["ticker", "date"]).copy()
    forward = (1.0 + ordered["return_1d"]).groupby(ordered["ticker"], sort=False).transform(
        lambda values: values.shift(-1)
        .iloc[::-1]
        .rolling(5, min_periods=5)
        .apply(np.prod, raw=True)
        .iloc[::-1]
        - 1.0
    )
    return forward.reindex(frame.index)


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
    """Add lifecycle metadata to a factor accepted into the persistent library."""
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
    payload["status"] = "active"
    payload["next_validation_at"] = (
        as_of + pd.Timedelta(days=interval_days)
    ).strftime("%Y-%m-%d")
    return payload


def _time_series_values(
    frame: pd.DataFrame, values: pd.Series, operation: str, window: int
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
    elif operation == "ts_delta":
        transformed = grouped.transform(lambda series: series - series.shift(window))
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
    if op in {"ts_mean", "ts_std", "ts_delta", "ts_zscore", "ts_rank"}:
        return _time_series_values(
            frame, frame[str(node["feature"])], op, int(node["window"])
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
    ranked: pd.DataFrame,
    frame: Optional[pd.DataFrame] = None,
    dates: Optional[pd.Series] = None,
) -> pd.Series:
    validate_candidate(candidate, ranked.columns)
    if candidate.expression_spec is not None:
        if frame is None or dates is None:
            raise ValueError("AST factors require the raw frame and dates for evaluation")
        validate_candidate(candidate, frame.columns)
        return _evaluate_expression(candidate.expression_spec, frame, dates)
    if candidate.op == "feature":
        return ranked[candidate.features[0]]
    left = ranked[candidate.features[0]]
    right = ranked[candidate.features[1]]
    if candidate.op == "difference":
        return left - right
    if candidate.op == "sum":
        return left + right
    raise ValueError(f"Unsupported operation: {candidate.op}")


def metric_dict(metric: Metric) -> Dict[str, float]:
    return asdict(metric)


def build_accepted_signal(
    frame: pd.DataFrame, ranked: pd.DataFrame, accepted: Sequence[Dict[str, object]]
) -> pd.DataFrame:
    """Build an equal-weight composite from validation-approved factors."""
    if not accepted:
        # A maintenance round may legitimately prune the entire library. Keep
        # an auditable empty signal instead of losing the registry snapshot.
        return pd.DataFrame(columns=["datetime", "instrument", "prediction"])
    scores = []
    for factor in accepted:
        candidate = Candidate(
            candidate_id=str(factor["candidate_id"]),
            op=str(factor["op"]),
            features=tuple(str(value) for value in factor["features"]),
            hypothesis=str(factor.get("hypothesis", "")),
        )
        direction = int(factor["direction"])
        scores.append(
            direction * evaluate_candidate_score(candidate, ranked, frame, frame["date"])
        )
    composite = pd.concat(scores, axis=1).mean(axis=1).rename("prediction")
    signal = pd.DataFrame(
        {
            "datetime": frame["date"].to_numpy(),
            "instrument": frame["ticker"].to_numpy(),
            "prediction": composite.to_numpy(dtype=float),
        }
    )
    return signal.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run the deterministic Miner-lite harness")
    parser.add_argument("--input", type=Path, default=base / "data" / "processed" / "sp500_features.csv.gz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_miner_lite",
    )
    parser.add_argument("--candidate-file", type=Path, default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Persistent factor-library registry; defaults to <output-dir>/registry.json",
    )
    parser.add_argument(
        "--maintenance-interval-days",
        type=int,
        default=90,
        help="Revalidate active factors after this many days (paper default: 90)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Maintenance date (YYYY-MM-DD); defaults to the latest available signal date",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Optional model; deepseek-chat/deepseek-reasoner use DEEPSEEK_API_KEY "
            "directly, other names use LiteLLM"
        ),
    )
    parser.add_argument("--llm-count", type=int, default=40)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument(
        "--max-temporal",
        type=int,
        default=20,
        help="Maximum deterministic candidates using the safe time-series DSL",
    )
    parser.add_argument("--min-train-ic", type=float, default=0.010)
    parser.add_argument("--min-valid-ic", type=float, default=0.005)
    parser.add_argument("--min-valid-hit", type=float, default=0.51)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.90,
        help="Minimum train/validation signal coverage (paper threshold: 0.90)",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=0.40,
        help="Maximum train/validation cross-sectional rank turnover (paper threshold: 0.40)",
    )
    args = parser.parse_args()
    if args.maintenance_interval_days < 1:
        parser.error("--maintenance-interval-days must be positive")
    registry_path = args.registry or args.output_dir / "registry.json"
    previous_registry = load_registry(registry_path)
    previous_value = previous_registry.get("accepted_factors", [])
    previous_factors = previous_value if isinstance(previous_value, list) else []

    frame = pd.read_csv(args.input, compression="infer", parse_dates=["date"])
    frame["target_5d"] = derive_target_5d(frame)
    frame["one_day_reversal"] = -frame["return_1d"]
    available = [column for column in frame.columns if column not in META_COLUMNS | BASELINE_FEATURES]
    # Keep rows whose 1-day label is valid even when the 5-day label is not
    # yet available at the end of the file; the horizon evaluator handles it.
    # Do not drop every feature here: candidate-level missing values must be
    # visible to the coverage metric.
    frame = frame.dropna(subset=["target_1d"]).reset_index(drop=True)
    dates = frame["date"]
    targets = {1: frame["target_1d"], 5: frame["target_5d"]}

    train_mask = dates.dt.year <= 2022
    valid_mask = dates.dt.year == 2023
    backtest_mask = dates.dt.year >= 2024

    if args.candidate_file:
        candidates = load_candidates(args.candidate_file, available)
        generator = "external_json"
    elif args.llm_model:
        candidates = generate_llm_candidates(
            args.llm_model,
            available,
            build_market_context(frame, train_mask),
            args.llm_count,
        )
        generator = f"llm:{args.llm_model}"
    else:
        candidates = template_candidates(available, args.max_pairs, args.max_temporal)
        generator = "template_v1"
    generated_candidate_count = len(candidates)

    # Retain valid factors from the previous library even if the current
    # generator does not propose them again.  This is the persistent Z_t layer.
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    invalid_previous_factors: List[str] = []
    for factor in previous_factors:
        if not isinstance(factor, dict):
            continue
        try:
            previous_candidate = candidate_from_factor(factor)
        except (TypeError, ValueError):
            invalid_previous_factors.append(str(factor.get("candidate_id", "<unknown>")))
            continue
        candidate_by_id.setdefault(previous_candidate.candidate_id, previous_candidate)
    candidates = list(candidate_by_id.values())
    previous_by_id = {
        str(factor["candidate_id"]): factor
        for factor in previous_factors
        if isinstance(factor, dict) and factor.get("candidate_id")
    }
    try:
        as_of = (
            pd.Timestamp(args.as_of).normalize()
            if args.as_of
            else dates.max().normalize()
        )
    except ValueError as exc:
        parser.error(f"Invalid --as-of date: {exc}")

    # Cross-sectional rank normalization is computed once for all source fields.
    ranked = frame[available].groupby(dates, sort=False).rank(pct=True).astype("float32")
    baseline = frame["one_day_reversal"].groupby(dates, sort=False).rank(pct=True)
    results: List[Dict[str, object]] = []
    accepted: List[Dict[str, object]] = []
    for candidate in candidates:
        score = evaluate_candidate_score(candidate, ranked, frame, dates)
        horizon_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for horizon in EVALUATION_HORIZONS:
            horizon_target = targets[horizon]
            horizon_metrics[str(horizon)] = {
                "train": metric_dict(
                    evaluate_score(dates, score, horizon_target, train_mask, frame["ticker"])
                ),
                "validation": metric_dict(
                    evaluate_score(dates, score, horizon_target, valid_mask, frame["ticker"])
                ),
                "backtest": metric_dict(
                    evaluate_score(dates, score, horizon_target, backtest_mask, frame["ticker"])
                ),
            }
        train = horizon_metrics["1"]["train"]
        valid = horizon_metrics["1"]["validation"]
        backtest = horizon_metrics["1"]["backtest"]
        direction = 1 if train["mean_daily_rank_ic"] >= 0 else -1
        directional_valid_ic = direction * valid["mean_daily_rank_ic"]
        directional_valid_hit = (
            valid["rank_ic_hit_ratio"] if direction > 0 else 1.0 - valid["rank_ic_hit_ratio"]
        )
        quality_metrics = [
            horizon_metrics[str(horizon)][split]
            for horizon in EVALUATION_HORIZONS
            for split in ("train", "validation")
        ]
        quality_ok = (
            all(metric["coverage"] >= args.min_coverage for metric in quality_metrics)
            and all(metric["turnover"] <= args.max_turnover for metric in quality_metrics)
        )
        accepted_flag = (
            abs(train["mean_daily_rank_ic"]) >= args.min_train_ic
            and directional_valid_ic >= args.min_valid_ic
            and directional_valid_hit >= args.min_valid_hit
            and quality_ok
        )
        row = {
            "candidate_id": candidate.candidate_id,
            "expression": candidate.expression,
            "op": candidate.op,
            "features": ",".join(candidate.features),
            "hypothesis": candidate.hypothesis,
            "direction_from_train": direction,
            "accepted": accepted_flag,
            # Keep one-day aliases for backwards-compatible consumers.
            "train": train,
            "validation": valid,
            "backtest": backtest,
            "horizons": horizon_metrics,
        }
        results.append(row)
        if accepted_flag:
            accepted.append(
                {
                    **asdict(candidate),
                    "features": list(candidate.features),
                    "expression": candidate.expression,
                    "direction_expression": (
                        candidate.expression if direction > 0 else f"-({candidate.expression})"
                    ),
                    "direction": direction,
                    "train_rank_ic": train["mean_daily_rank_ic"],
                    "validation_rank_ic": valid["mean_daily_rank_ic"],
                    "validation_directional_rank_ic": directional_valid_ic,
                    "validation_directional_hit_ratio": directional_valid_hit,
                    "backtest_directional_rank_ic": direction * backtest["mean_daily_rank_ic"],
                    "backtest_rank_ic": backtest["mean_daily_rank_ic"],
                    "horizon_metrics": horizon_metrics,
                }
            )

    current_accepted = accepted
    current_accepted_by_id = {
        str(factor["candidate_id"]): factor for factor in current_accepted
    }
    active_factors: List[Dict[str, object]] = []
    retained_without_revalidation: List[str] = []
    revalidated_factors: List[str] = []
    pruned_factors: List[str] = []
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        previous = previous_by_id.get(candidate_id)
        if previous and not needs_revalidation(previous, as_of, args.maintenance_interval_days):
            carried = dict(previous)
            carried["status"] = "active"
            carried["revalidation_due"] = False
            if not carried.get("next_validation_at"):
                last_validated = pd.Timestamp(str(carried["last_validated_at"]))
                carried["next_validation_at"] = (
                    last_validated + pd.Timedelta(days=args.maintenance_interval_days)
                ).strftime("%Y-%m-%d")
            active_factors.append(carried)
            retained_without_revalidation.append(candidate_id)
        elif candidate_id in current_accepted_by_id:
            active_factors.append(
                library_factor(
                    current_accepted_by_id[candidate_id],
                    previous,
                    as_of,
                    args.maintenance_interval_days,
                )
            )
            if previous:
                revalidated_factors.append(candidate_id)
        elif previous:
            pruned_factors.append(candidate_id)

    accepted = active_factors
    accepted.sort(
        key=lambda item: float(item.get("validation_directional_rank_ic", float("-inf"))),
        reverse=True,
    )
    maintenance = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "interval_days": args.maintenance_interval_days,
        "generated_candidate_count": generated_candidate_count,
        "candidate_count_including_library": len(candidates),
        "previous_active_count": len(previous_factors),
        "current_round_gate_pass_count": len(current_accepted),
        "new_or_revalidated_count": len(
            [factor for factor in current_accepted if str(factor["candidate_id"]) not in previous_by_id]
        ) + len(revalidated_factors),
        "active_count": len(accepted),
        "retained_without_revalidation": retained_without_revalidation,
        "revalidated": revalidated_factors,
        "pruned": pruned_factors,
        "invalid_previous_factors": invalid_previous_factors,
    }
    baseline_horizons: Dict[str, Dict[str, Dict[str, float]]] = {}
    for horizon in EVALUATION_HORIZONS:
        horizon_target = targets[horizon]
        baseline_horizons[str(horizon)] = {
            "train": metric_dict(
                evaluate_score(dates, baseline, horizon_target, train_mask, frame["ticker"])
            ),
            "validation": metric_dict(
                evaluate_score(dates, baseline, horizon_target, valid_mask, frame["ticker"])
            ),
            "backtest": metric_dict(
                evaluate_score(dates, baseline, horizon_target, backtest_mask, frame["ticker"])
            ),
        }
    baseline_metrics = baseline_horizons["1"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    factor_library_path = args.output_dir / "factor_library.json"
    signal = build_accepted_signal(frame, ranked, accepted)
    signal.to_csv(args.output_dir / "predictions.csv", index=False)
    candidate_payload = []
    for candidate in candidates:
        payload = asdict(candidate)
        payload["features"] = list(candidate.features)
        if candidate.expression_spec is not None:
            payload["expression"] = candidate.expression_spec
            payload["expression_text"] = candidate.expression
        else:
            payload["expression"] = candidate.expression
        candidate_payload.append(payload)
    (args.output_dir / "candidates.json").write_text(
        json.dumps(candidate_payload, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "candidate_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    (args.output_dir / "accepted_factors.json").write_text(
        json.dumps(accepted, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    factor_library_path.write_text(
        json.dumps(
            {
                "library_version": "miner-factor-library-v1",
                "as_of": maintenance["as_of"],
                "maintenance": maintenance,
                "factors": accepted,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n"
    )
    registry_path.write_text(
        json.dumps(
            {
                "registry_version": "miner-lite-v2",
                "generator": generator,
                "input": str(args.input),
                "as_of": maintenance["as_of"],
                "generated_candidate_count": generated_candidate_count,
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "factor_library_file": factor_library_path.name,
                "signal": {
                    "file": "predictions.csv",
                    "construction": "equal-weight mean of direction-adjusted accepted factor scores",
                    "execution": "T close signal, T+1 open trade",
                },
                "selection": {
                    "train_end": "2022-12-30",
                    "validation": "2023",
                    "holdout": "2024-2026",
                    "min_train_abs_rank_ic": args.min_train_ic,
                    "min_validation_directional_rank_ic": args.min_valid_ic,
                    "min_validation_hit_ratio": args.min_valid_hit,
                    "min_coverage": args.min_coverage,
                    "max_turnover": args.max_turnover,
                    "evaluation_horizons": list(EVALUATION_HORIZONS),
                },
                "baseline": baseline_metrics,
                "baseline_horizons": baseline_horizons,
                "maintenance": maintenance,
                "accepted_factors": accepted,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n"
    )

    metrics_rows: List[Dict[str, object]] = []
    for result in results:
        for horizon, horizon_metrics in result["horizons"].items():
            for split in ("train", "validation", "backtest"):
                metrics_rows.append(
                    {
                        "candidate_id": result["candidate_id"],
                        "expression": result["expression"],
                        "accepted": result["accepted"],
                        "horizon": int(horizon),
                        "split": split,
                        **horizon_metrics[split],
                    }
                )
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "candidate_metrics.csv", index=False)

    report_lines = [
        "# AlphaCrafter Miner-lite 报告",
        "",
        f"- 候选生成器：`{generator}`",
        f"- 本轮生成候选数：`{generated_candidate_count}`",
        f"- 合并因子库后候选数：`{len(candidates)}`",
        f"- 本轮准入门通过数：`{len(current_accepted)}`",
        f"- 当前活跃因子数：`{len(accepted)}`",
        "- 训练集：2016–2022；验证集：2023；留出回测：2024–2026",
        "- `return_1d` 与 `one_day_reversal` 不进入候选池，避免重复计算简单反转因子。",
        "",
        "## 基线",
        "",
        "| split | Rank IC | Rank ICIR | Hit Ratio | Coverage | Turnover | Top/Bottom spread |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "backtest"):
        metric = baseline_metrics[split]
        report_lines.append(
            f"| {split} | {metric['mean_daily_rank_ic']:.6f} | {metric['rank_icir']:.4f} | "
            f"{metric['rank_ic_hit_ratio']:.3f} | {metric['coverage']:.3f} | "
            f"{metric['turnover']:.3f} | {metric['mean_top_bottom_decile_spread']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## 多周期质量诊断（基线）",
            "",
            "| horizon | split | Rank IC | Rank ICIR | Coverage | Turnover |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in EVALUATION_HORIZONS:
        for split in ("train", "validation", "backtest"):
            metric = baseline_horizons[str(horizon)][split]
            report_lines.append(
                f"| {horizon}d | {split} | {metric['mean_daily_rank_ic']:.6f} | "
                f"{metric['rank_icir']:.4f} | {metric['coverage']:.3f} | {metric['turnover']:.3f} |"
            )
    report_lines.extend(["", "## 通过验证的因子（按验证集方向调整后的 Rank IC 排序）", ""])
    if accepted:
        report_lines.extend([
            "| 因子 | 表达式 | 训练 Rank IC | 验证 Rank IC | 回测 Rank IC |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for factor in accepted[:20]:
            report_lines.append(
                f"| `{factor['candidate_id']}` | `{factor['direction_expression']}` | "
                f"{factor['train_rank_ic']:.6f} | {factor['validation_directional_rank_ic']:.6f} | "
                f"{factor['backtest_directional_rank_ic']:.6f} |"
            )
    else:
        report_lines.append("没有候选因子通过当前训练集和验证集门槛。")
    report_lines.extend(
        [
            "",
            "> 验证集用于因子准入；2024–2026 只作为最终留出报告，不能用于调阈值。",
            f"> 多周期评估：{', '.join(str(horizon) + 'd' for horizon in EVALUATION_HORIZONS)}；覆盖率最低 {args.min_coverage:.2f}，排名换手率最高 {args.max_turnover:.2f}。",
            f"> 因子库维护：截至 {maintenance['as_of']}，间隔 {args.maintenance_interval_days} 天；本轮淘汰 {len(pruned_factors)} 个因子。",
            "> 当前版本是 Miner-lite：候选可由安全模板或 MinerLLM 生成，也支持通过 `--candidate-file` 接收外部 JSON。",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(f"Generated candidates: {len(candidates)}")
    print(f"Accepted factors: {len(accepted)}")
    print(f"Results written to {args.output_dir}")
    for factor in accepted[:10]:
        print(
            f"{factor['candidate_id']}: valid={factor['validation_directional_rank_ic']:.6f}, "
            f"backtest={factor['backtest_directional_rank_ic']:.6f}"
        )


if __name__ == "__main__":
    main()
