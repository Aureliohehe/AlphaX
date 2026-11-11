#!/usr/bin/env python3
"""AlphaCrafter Screener-lite.

Select an ensemble from the persistent Miner factor library using only
information available before each signal date.  The implementation follows the
paper's Screener policy in a deterministic form:

1. Compute each factor's trailing 10-session cross-sectional Rank IC.
2. Keep factors above a suitability threshold.
3. Remove structurally redundant factors with a lightweight expression
   similarity check.
4. Weight the survivors by absolute suitability and use the current IC sign as
   their direction.
5. Emit a time-varying composite signal suitable for the Qlib backtester.

The paper uses an LLM for regime and semantic relevance judgments.  This
lite version keeps those decisions deterministic and auditable; regime
statistics are still recorded for later comparison with an LLM Screener.
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from temporal import (
    active_factors_at, label_end_dates, load_timeline, session_date, snapshot_date, visible_panel,
)

from miner_lite import (
    BASELINE_FEATURES,
    META_COLUMNS,
    Candidate,
    _deepseek_completion,
    _deepseek_tool_call,
    candidate_from_factor,
    evaluate_candidate_score,
)


def load_factor_library(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Factor library must be a JSON object")
    factors = payload.get("factors", payload.get("accepted_factors", []))
    if not isinstance(factors, list):
        raise ValueError("Factor library must contain factors or accepted_factors")
    active = []
    for factor in factors:
        if not isinstance(factor, dict):
            continue
        if factor.get("status", "active") != "active":
            continue
        active.append(factor)
    return active


def cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.groupby(dates, sort=False).rank(pct=True) - 0.5


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 3 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return float("nan")
    left_rank = valid["left"].rank(method="average").to_numpy(dtype=float)
    right_rank = valid["right"].rank(method="average").to_numpy(dtype=float)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.sqrt(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank))
    return float(np.dot(left_rank, right_rank) / denominator) if denominator else float("nan")


def daily_rank_ic(
    dates: pd.Series, score: pd.Series, target: pd.Series
) -> pd.Series:
    values = pd.DataFrame({"date": dates, "score": score, "target": target}).dropna()
    result = {}
    for date, group in values.groupby("date", sort=True):
        result[date] = rank_correlation(group["score"], group["target"])
    return pd.Series(result, dtype=float).sort_index()


def expression_similarity(left: Candidate, right: Candidate) -> float:
    """Approximate semantic overlap without requiring another LLM call."""
    if left.expression == right.expression:
        return 1.0
    left_features = set(left.features)
    right_features = set(right.features)
    union = left_features | right_features
    feature_overlap = len(left_features & right_features) / len(union) if union else 0.0
    text_similarity = SequenceMatcher(None, left.expression, right.expression).ratio()
    op_similarity = 1.0 if left.op == right.op else 0.0
    if feature_overlap == 0.0:
        # Similar spelling alone is not semantic overlap: return_5d and
        # return_20d should not be treated as identical just because they
        # share the same operator text.
        return 0.5 * op_similarity
    return max(feature_overlap, 0.5 * text_similarity + 0.5 * op_similarity)


def market_regime(frame: pd.DataFrame, date: pd.Timestamp) -> Dict[str, object]:
    """Describe the observable market state at a signal date."""
    current = frame[frame["date"] == date]
    history = frame[frame["date"] < date]
    median_return = float(current["return_1d"].median()) if not current.empty else float("nan")
    breadth = float((current["return_1d"] > 0).mean()) if not current.empty else float("nan")
    current_vol = float(current["volatility_20d"].median()) if "volatility_20d" in current else float("nan")
    history_vol = (
        float(history["volatility_20d"].median())
        if "volatility_20d" in history and not history.empty
        else float("nan")
    )
    if not np.isnan(current_vol) and not np.isnan(history_vol) and current_vol > 1.25 * history_vol:
        regime = "high_volatility"
    elif median_return >= 0.001 and breadth >= 0.55:
        regime = "bullish_trend"
    elif median_return <= -0.001 and breadth <= 0.45:
        regime = "bearish_trend"
    else:
        regime = "mixed_or_sideways"
    return {
        "regime": regime,
        "median_return_1d": median_return,
        "positive_return_breadth": breadth,
        "median_volatility_20d": current_vol,
    }


FACTOR_RELEVANCE_TOOL: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "judge_factor_relevance",
        "description": "Judge whether a factor is relevant to the supplied market regime.",
        "parameters": {
            "type": "object",
            "properties": {
                "relevant": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["relevant", "confidence", "reason"],
            "additionalProperties": False,
        },
    },
}


def judge_factor_relevance_with_tool(
    candidate: Candidate, regime: Dict[str, object], model: str
) -> Dict[str, object]:
    """Use one typed tool call to judge a factor; selection is not changed yet."""
    model_name = model.split("/", 1)[1] if model.startswith("deepseek/") else model
    if model_name not in {"deepseek-chat", "deepseek-reasoner"}:
        raise ValueError("Factor relevance tool currently supports DeepSeek models only")
    messages: List[Dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "You are a factor relevance judge. Use only the supplied regime and factor "
                "description. Do not calculate returns or make a trade decision."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "regime": regime,
                    "factor_id": candidate.candidate_id,
                    "factor_expression": candidate.expression,
                    "features": list(candidate.features),
                    "hypothesis": candidate.hypothesis,
                },
                ensure_ascii=False,
            ),
        },
    ]
    result = _deepseek_tool_call(messages, model_name, FACTOR_RELEVANCE_TOOL)
    if not isinstance(result.get("relevant"), bool):
        raise ValueError("Tool result 'relevant' must be boolean")
    confidence = float(result.get("confidence", -1))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Tool result confidence must be between 0 and 1")
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Tool result reason must be non-empty text")
    return {"relevant": result["relevant"], "confidence": confidence, "reason": reason}


FACTOR_SIMILARITY_TOOL: Dict[str, object] = {
    "type": "function",
    "function": {
        "name": "judge_factor_similarity",
        "description": "Judge semantic redundancy between two factor expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "similarity": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["similarity", "reason"],
            "additionalProperties": False,
        },
    },
}


def judge_factor_similarity_with_tool(
    left: Candidate, right: Candidate, model: str
) -> Dict[str, object]:
    """Use one typed tool call to assess semantic redundancy."""
    model_name = model.split("/", 1)[1] if model.startswith("deepseek/") else model
    if model_name not in {"deepseek-chat", "deepseek-reasoner"}:
        raise ValueError("Factor similarity tool currently supports DeepSeek models only")
    result = _deepseek_tool_call(
        [
            {
                "role": "system",
                "content": "Compare factor semantics only. Do not use returns or make a trade decision.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "factor_a": {"expression": left.expression, "features": list(left.features)},
                        "factor_b": {"expression": right.expression, "features": list(right.features)},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        model_name,
        FACTOR_SIMILARITY_TOOL,
    )
    try:
        similarity = float(result.get("similarity", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Tool result similarity must be numeric") from exc
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("Tool result similarity must be between 0 and 1")
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Tool result reason must be non-empty text")
    return {"similarity": similarity, "reason": reason}


def assess_regime_with_llm(
    frame: pd.DataFrame, date: pd.Timestamp, model: str
) -> Dict[str, object]:
    """Ask DeepSeek for a structured regime diagnosis at one signal date."""
    history = frame[frame["date"] <= date].sort_values("date")
    recent_dates = history["date"].drop_duplicates().tail(60)
    recent = history[history["date"].isin(recent_dates)]
    fields = ["return_1d", "return_5d", "return_20d", "volatility_20d", "volume_ratio_20d"]
    summary = {
        field: float(recent[field].median())
        for field in fields
        if field in recent and recent[field].notna().any()
    }
    breadth = float((recent["return_1d"] > 0).mean()) if "return_1d" in recent else None
    prompt = f"""Assess the market regime for a cross-sectional equity factor screener.
Return only one JSON object with this schema:
{{
  "regime": "bullish_trend|bearish_trend|high_volatility|mixed_or_sideways",
  "relevant_themes": ["momentum|reversal|volatility|volume"],
  "confidence": 0.0,
  "rationale": "one short sentence"
}}
Use only the supplied point-in-time statistics. Do not make a trade decision.
As-of date: {date.date()}
Recent 60-session full-universe statistics: {json.dumps(summary, ensure_ascii=False)}
Recent positive-return breadth: {breadth}
"""
    model_name = model.split("/", 1)[1] if model.startswith("deepseek/") else model
    if model_name not in {"deepseek-chat", "deepseek-reasoner"}:
        raise ValueError("LLM regime judge currently supports deepseek-chat/deepseek-reasoner only")
    content = _deepseek_completion(
        [
            {"role": "system", "content": "Return only valid JSON matching the supplied schema."},
            {"role": "user", "content": prompt},
        ],
        model_name,
    ).strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Regime response must be a JSON object")
    allowed_regimes = {"bullish_trend", "bearish_trend", "high_volatility", "mixed_or_sideways"}
    if result.get("regime") not in allowed_regimes:
        raise ValueError("Regime response contains an unsupported regime")
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Regime confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Regime confidence must be between 0 and 1")
    result["confidence"] = confidence
    result["source"] = "llm"
    return result


def select_ensemble(
    date: pd.Timestamp,
    dates: Sequence[pd.Timestamp],
    candidates: Dict[str, Candidate],
    daily_ics: Dict[str, pd.Series],
    min_history_days: int,
    min_suitability: float,
    similarity_threshold: float,
    max_factors: int,
    relevance: Optional[Dict[str, Dict[str, object]]] = None,
    semantic_similarity: Optional[Dict[Tuple[str, str], Dict[str, object]]] = None,
    label_available_dates: Optional[pd.Series] = None,
) -> List[Dict[str, object]]:
    position = list(dates).index(date)
    previous_dates = list(dates[max(0, position - min_history_days) : position])
    if len(previous_dates) < min_history_days:
        return []
    if label_available_dates is not None:
        maturity = label_available_dates.reindex(previous_dates)
        if maturity.isna().any() or (maturity > date).any():
            return []

    scored = []
    for candidate_id, ics in daily_ics.items():
        if candidate_id not in candidates:
            continue
        if relevance is not None:
            relevance_result = relevance.get(candidate_id)
            if relevance_result is not None and not bool(relevance_result["relevant"]):
                continue
        trailing = ics.reindex(previous_dates).dropna()
        if len(trailing) < min_history_days:
            continue
        suitability = float(trailing.mean())
        if abs(suitability) >= min_suitability:
            scored.append((candidate_id, suitability))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)

    selected: List[Tuple[str, float]] = []
    for candidate_id, suitability in scored:
        candidate = candidates[candidate_id]
        def is_redundant(selected_id: str) -> bool:
            pair = tuple(sorted((candidate_id, selected_id)))
            if semantic_similarity is not None and pair in semantic_similarity:
                similarity = float(semantic_similarity[pair]["similarity"])
            else:
                similarity = expression_similarity(candidate, candidates[selected_id])
            return similarity >= similarity_threshold

        if any(is_redundant(selected_id) for selected_id, _ in selected):
            continue
        selected.append((candidate_id, suitability))
        if len(selected) >= max_factors:
            break

    total = sum(abs(suitability) for _, suitability in selected)
    return [
        {
            "candidate_id": candidate_id,
            "suitability_rank_ic": suitability,
            "direction": 1 if suitability >= 0 else -1,
            "weight": abs(suitability) / total if total else 0.0,
            "expression": candidates[candidate_id].expression,
        }
        for candidate_id, suitability in selected
    ]


def ensemble_signal(
    rank_scores: Dict[str, pd.Series],
    ensemble: Sequence[Dict[str, object]],
    index: pd.Index,
) -> pd.Series:
    """Missing component signals remain missing, independent of factor order."""
    if not ensemble:
        return pd.Series(np.nan, index=index, dtype=float)
    components = pd.concat(
        [rank_scores[factor["candidate_id"]].reindex(index) * factor["weight"] * factor["direction"]
         for factor in ensemble], axis=1,
    )
    return components.sum(axis=1, min_count=len(ensemble))


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run the deterministic AlphaCrafter Screener-lite")
    parser.add_argument(
        "--input", type=Path, default=base / "data" / "processed" / "sp500_features.csv.gz"
    )
    library_source = parser.add_mutually_exclusive_group()
    library_source.add_argument(
        "--factor-library",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_miner_lite" / "factor_library.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_screener_lite",
    )
    library_source.add_argument("--library-history", type=Path, help="Directory of write-once Miner snapshots for point-in-time replay")
    parser.add_argument("--allow-retrospective-library", action="store_true", help="DIAGNOSTIC ONLY: allow a static legacy/future library on all historical dates")
    parser.add_argument("--as-of", default=None, help="Last visible feature session; defaults to input maximum date")
    parser.add_argument("--signal-start", default=None, help="First output signal date; earlier feature/IC history is retained")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument(
        "--min-suitability",
        type=float,
        default=0.02,
        help="Minimum trailing Rank IC suitability (paper threshold: 0.02)",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    parser.add_argument("--max-factors", type=int, default=20)
    parser.add_argument(
        "--llm-regime-model",
        default=None,
        help="Optional DeepSeek model for one latest-date regime diagnosis",
    )
    parser.add_argument(
        "--llm-relevance-model",
        default=None,
        help="Optional DeepSeek model for latest-date factor relevance tool calls",
    )
    parser.add_argument(
        "--llm-similarity-model",
        default=None,
        help="Optional DeepSeek model for latest-date factor similarity tool calls",
    )
    args = parser.parse_args()
    if args.lookback_days < 1 or args.max_factors < 1:
        parser.error("lookback and max-factors must be positive")

    frame = pd.read_csv(args.input, compression="infer", parse_dates=["date"])
    execution_target = "target_open_close" if "target_open_close" in frame else "target_1d"
    # Forward-label availability is not observable at signal time. Retain the
    # complete feature universe, including the latest unlabelled session.
    as_of = session_date(args.as_of if args.as_of else frame["date"].max())
    frame = visible_panel(frame, as_of)
    label_ends = label_end_dates(frame, 1)
    frame.loc[label_ends.isna(), execution_target] = np.nan
    label_available_dates = label_ends.groupby(frame["date"]).max()
    # A truncated/delisted instrument may have an unknown next quote. Do not
    # compute a deceptively 'complete' IC from the other instruments alone.
    unknown_end = label_ends.isna().groupby(frame["date"]).any()
    label_available_dates = label_available_dates.mask(unknown_end)
    available = [column for column in frame.columns if column not in META_COLUMNS | BASELINE_FEATURES]
    ranked = frame[available].groupby(frame["date"], sort=False).rank(pct=True).astype("float32")
    dates = sorted(frame["date"].drop_duplicates())
    signal_start = session_date(args.signal_start) if args.signal_start else dates[0]
    signal_dates = [date for date in dates if date >= signal_start]
    if not signal_dates:
        parser.error("No visible sessions on or after --signal-start")
    if args.library_history:
        if args.allow_retrospective_library:
            parser.error("Retrospective diagnostics cannot be combined with snapshot replay")
        timeline = load_timeline(args.library_history)
    else:
        payload = json.loads(args.factor_library.read_text())
        if args.allow_retrospective_library:
            timeline = [(dates[0], {"factors": load_factor_library(args.factor_library)})]
        else:
            timeline = [(snapshot_date(payload), payload)]
    # Future definitions must not be evaluated or passed to the LLM at this cutoff.
    factors = [factor for cutoff, payload in timeline if cutoff <= as_of
               for factor in payload.get("factors", payload.get("accepted_factors", []))]
    candidates: Dict[str, Candidate] = {}
    invalid_factors = []
    for factor in factors:
        try:
            candidate = candidate_from_factor(factor)
        except (TypeError, ValueError) as exc:
            invalid_factors.append({"candidate_id": factor.get("candidate_id", "<unknown>"), "error": str(exc)})
            continue
        previous = candidates.get(candidate.candidate_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"Factor ID {candidate.candidate_id} changes definition across snapshots; use versioned IDs")
        candidates[candidate.candidate_id] = candidate
    if invalid_factors and not args.allow_retrospective_library:
        raise ValueError(f"Invalid snapshot factors: {invalid_factors}")
    latest_ids = {factor["candidate_id"] for factor in active_factors_at(timeline, dates[-1])}
    latest_candidates = {key: value for key, value in candidates.items() if key in latest_ids}

    latest_llm_regime: Optional[Dict[str, object]] = None
    latest_deterministic_regime = market_regime(frame, dates[-1])
    if args.llm_regime_model:
        try:
            latest_llm_regime = assess_regime_with_llm(frame, dates[-1], args.llm_regime_model)
        except (ValueError, RuntimeError, SystemExit) as exc:
            raise SystemExit(f"LLM regime judge failed: {exc}") from exc

    latest_relevance: Optional[Dict[str, Dict[str, object]]] = None
    if args.llm_relevance_model:
        relevance_regime = latest_llm_regime or latest_deterministic_regime
        latest_relevance = {}
        for candidate_id, candidate in latest_candidates.items():
            try:
                latest_relevance[candidate_id] = judge_factor_relevance_with_tool(
                    candidate, relevance_regime, args.llm_relevance_model
                )
            except (ValueError, RuntimeError, SystemExit) as exc:
                raise SystemExit(
                    f"LLM relevance judge failed for {candidate_id}: {exc}"
                ) from exc

    scores: Dict[str, pd.Series] = {}
    rank_scores: Dict[str, pd.Series] = {}
    daily_ics: Dict[str, pd.Series] = {}
    for candidate_id, candidate in candidates.items():
        score = evaluate_candidate_score(candidate, ranked, frame, frame["date"])
        scores[candidate_id] = score
        rank_scores[candidate_id] = cross_section_rank(score, frame["date"])
        daily_ics[candidate_id] = daily_rank_ic(frame["date"], score, frame[execution_target])

    latest_similarity: Optional[Dict[Tuple[str, str], Dict[str, object]]] = None
    if args.llm_similarity_model:
        latest_similarity = {}
        candidate_items = list(latest_candidates.items())
        for left_index, (left_id, left_candidate) in enumerate(candidate_items):
            for right_id, right_candidate in candidate_items[left_index + 1 :]:
                try:
                    latest_similarity[tuple(sorted((left_id, right_id)))] = judge_factor_similarity_with_tool(
                        left_candidate, right_candidate, args.llm_similarity_model
                    )
                except (ValueError, RuntimeError, SystemExit) as exc:
                    raise SystemExit(
                        f"LLM similarity judge failed for {left_id}/{right_id}: {exc}"
                    ) from exc

    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    selection_rows: List[Dict[str, object]] = []
    regime_rows: List[Dict[str, object]] = []
    selected_days = 0
    for date in signal_dates:
        active_ids = {factor["candidate_id"] for factor in active_factors_at(timeline, date)}
        active_candidates = {key: value for key, value in candidates.items() if key in active_ids}
        ensemble = select_ensemble(
            date,
            dates,
            active_candidates,
            daily_ics,
            args.lookback_days,
            args.min_suitability,
            args.similarity_threshold,
            args.max_factors,
            latest_relevance if date == dates[-1] else None,
            latest_similarity if date == dates[-1] else None,
            label_available_dates=label_available_dates,
        )
        regime = market_regime(frame, date)
        regime["regime_source"] = "deterministic"
        if latest_llm_regime is not None and date == dates[-1]:
            regime.update(
                {
                    "llm_regime": latest_llm_regime["regime"],
                    "llm_relevant_themes": latest_llm_regime.get("relevant_themes", []),
                    "llm_confidence": latest_llm_regime["confidence"],
                    "llm_rationale": latest_llm_regime.get("rationale", ""),
                    "regime_source": "llm",
                }
            )
        regime_rows.append({"date": str(date.date()), **regime, "active_factor_count": len(active_candidates), "selected_factor_count": len(ensemble)})
        if not ensemble:
            continue
        selected_days += 1
        day_mask = frame["date"] == date
        for factor in ensemble:
            factor_row = {"date": str(date.date()), **factor, **regime}
            if latest_relevance is not None and date == dates[-1]:
                factor_row.update(
                    {
                        "llm_relevant": latest_relevance[factor["candidate_id"]]["relevant"],
                        "llm_relevance_confidence": latest_relevance[factor["candidate_id"]]["confidence"],
                        "llm_relevance_reason": latest_relevance[factor["candidate_id"]]["reason"],
                    }
                )
            selection_rows.append(factor_row)
        predictions.loc[day_mask] = ensemble_signal(rank_scores, ensemble, frame.index[day_mask])

    signal = pd.DataFrame(
        {
            "datetime": frame.loc[predictions.notna(), "date"].to_numpy(),
            "instrument": frame.loc[predictions.notna(), "ticker"].to_numpy(),
            "prediction": predictions.loc[predictions.notna()].to_numpy(dtype=float),
        }
    ).sort_values(["datetime", "instrument"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.to_csv(args.output_dir / "predictions.csv", index=False)
    selection_frame = pd.DataFrame(selection_rows) if selection_rows else pd.DataFrame(columns=["date", "candidate_id", "weight", "direction"])
    selection_frame.to_csv(args.output_dir / "selection.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(args.output_dir / "regime.csv", index=False)

    summary = {
        "screener_version": "screener-lite-v3",
        "as_of": str(as_of.date()),
        "signal_start": str(signal_start.date()),
        "output_session_count": len(signal_dates),
        "library_scope": "retrospective_diagnostic" if args.allow_retrospective_library else "point_in_time_snapshots",
        "point_in_time_warning": "Retrospective diagnostics are not out-of-sample evidence" if args.allow_retrospective_library else "Snapshot cutoff enforced; source data and external candidate provenance still require independent audit",
        "library_history": str(args.library_history) if args.library_history else None,
        "snapshot_dates": [str(cutoff.date()) for cutoff, _ in timeline],
        "factor_library": str(args.factor_library),
        "input": str(args.input),
        "factor_count": len(candidates),
        "selected_signal_days": selected_days,
        "lookback_days": args.lookback_days,
        "min_suitability_rank_ic": args.min_suitability,
        "similarity_threshold": args.similarity_threshold,
        "max_factors": args.max_factors,
        "llm_regime_model": args.llm_regime_model,
        "llm_regime_scope": "latest available signal date only" if args.llm_regime_model else None,
        "llm_relevance_model": args.llm_relevance_model,
        "llm_relevance_scope": "latest available signal date only" if args.llm_relevance_model else None,
        "llm_similarity_model": args.llm_similarity_model,
        "llm_similarity_scope": "latest available signal date only" if args.llm_similarity_model else None,
        "llm_similarity_pairs": (
            {"|||".join(pair): result for pair, result in latest_similarity.items()}
            if latest_similarity is not None
            else None
        ),
        "semantic_filter": "LLM latest-date similarity" if args.llm_similarity_model else "deterministic expression/feature similarity",
        "signal": {
            "file": "predictions.csv",
            "construction": "weighted mean of direction-adjusted cross-sectional factor ranks",
            "execution": "T close signal, T+1 open trade",
        },
        "invalid_factors": invalid_factors,
    }
    (args.output_dir / "registry.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    report = [
        "# AlphaCrafter Screener-lite 报告",
        "",
        f"- 历史因子定义数：`{len(candidates)}`；信息截止：`{as_of.date()}`",
        f"- 因子库模式：`{summary['library_scope']}`；快照按 as-of 激活，空快照清空因子库。",
        f"- 产生信号的交易日：`{selected_days}` / `{len(signal_dates)}`",
        f"- 滚动窗口：`{args.lookback_days}` 日",
        f"- Suitability 阈值：`{args.min_suitability}`",
        f"- 语义相似度阈值：`{args.similarity_threshold}`",
        "",
        f"> 市场状态：{'已启用最新日期的 LLM regime judge' if args.llm_regime_model else '确定性摘要'}；因子相关性：{'已启用最新日期的 Tool Calling' if args.llm_relevance_model else '确定性筛选'}；语义去重：{'已启用最新日期的 Tool Calling' if args.llm_similarity_model else '确定性相似度'}。",
        "> 每个交易日的因子适用性只使用此前已产生标签的交易日，避免筛选时使用当日未来收益。",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Active factors: {len(candidates)}")
    print(f"Signal days: {selected_days}/{len(signal_dates)}")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
