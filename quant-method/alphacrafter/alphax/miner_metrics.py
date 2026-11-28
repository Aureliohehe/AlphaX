"""Shared factor metrics, forward-return labels, and Miner admission checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EVALUATION_HORIZONS = (1, 5)
# Thresholds from AlphaCrafter Algorithm 1.  They apply to both the
# training and validation windows used by this implementation.
PAPER_HORIZON_THRESHOLDS = {
    1: {"ic": 0.015, "rank_ic": 0.015, "icir": 0.20, "rank_icir": 0.20},
    5: {"ic": 0.025, "rank_ic": 0.025, "icir": 0.25, "rank_icir": 0.25},
}
PAPER_MIN_HIT_DEVIATION = 0.10


@dataclass
class Metric:
    rows: int
    mean_daily_ic: float
    daily_ic_std: float
    icir: float
    mean_daily_rank_ic: float
    daily_rank_ic_std: float
    rank_icir: float
    rank_ic_hit_ratio: float
    mean_top_bottom_decile_spread: float
    coverage: float
    turnover: float


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
    daily_rank_ic: List[float] = []
    spreads: List[float] = []
    for _, group in values.groupby("date", sort=False):
        if len(group) < 3 or group["score"].nunique() < 2 or group["target"].nunique() < 2:
            continue
        # Pearson IC on the original values, as required by the paper.
        score_values = group["score"].to_numpy(dtype=float)
        target_values = group["target"].to_numpy(dtype=float)
        score_values -= score_values.mean()
        target_values -= target_values.mean()
        denominator = np.sqrt(np.dot(score_values, score_values) * np.dot(target_values, target_values))
        if denominator == 0:
            continue
        daily_ic.append(float(np.dot(score_values, target_values) / denominator))

        # Rank the score and target inside each trading day.  This is
        # equivalent to Spearman correlation and avoids scipy as a dependency.
        score_rank = group["score"].rank(method="average").to_numpy(dtype=float)
        target_rank = group["target"].rank(method="average").to_numpy(dtype=float)
        score_rank -= score_rank.mean()
        target_rank -= target_rank.mean()
        rank_denominator = np.sqrt(np.dot(score_rank, score_rank) * np.dot(target_rank, target_rank))
        if rank_denominator == 0:
            continue
        daily_rank_ic.append(float(np.dot(score_rank, target_rank) / rank_denominator))
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
    if len(dates_sorted) < 2:
        turnover = np.nan
    else:
        previous_dates = dict(zip(dates_sorted[1:], dates_sorted[:-1]))
        ranked["previous_date"] = pd.to_datetime(ranked["date"].map(previous_dates))
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
    rank_ic = pd.Series(daily_rank_ic, dtype=float).dropna()
    spread = pd.Series(spreads, dtype=float).dropna()
    if ic.empty or rank_ic.empty:
        return Metric(
            rows=0,
            mean_daily_ic=np.nan,
            daily_ic_std=np.nan,
            icir=np.nan,
            mean_daily_rank_ic=np.nan,
            daily_rank_ic_std=np.nan,
            rank_icir=np.nan,
            rank_ic_hit_ratio=np.nan,
            mean_top_bottom_decile_spread=np.nan,
            coverage=coverage,
            turnover=turnover,
        )
    # Paper Appendix B.2 defines sigma with 1/T (population variance).
    ic_std = float(ic.std(ddof=0))
    rank_ic_std = float(rank_ic.std(ddof=0))
    return Metric(
        rows=int(len(values)),
        mean_daily_ic=float(ic.mean()),
        daily_ic_std=ic_std,
        icir=float(ic.mean() / ic_std) if ic_std else np.nan,
        mean_daily_rank_ic=float(rank_ic.mean()),
        daily_rank_ic_std=rank_ic_std,
        rank_icir=float(rank_ic.mean() / rank_ic_std) if rank_ic_std else np.nan,
        rank_ic_hit_ratio=float((rank_ic > 0).mean()),
        mean_top_bottom_decile_spread=float(spread.mean()) if not spread.empty else np.nan,
        coverage=coverage,
        turnover=turnover,
    )


def derive_target_5d(frame: pd.DataFrame) -> pd.Series:
    """Derive a five-session close-to-close forward return for legacy files."""
    if "target_5d" in frame:
        return frame["target_5d"]
    if "label_end_5d" in frame:
        raise ValueError("Explicit-date panels must supply target_5d; cannot reconstruct raw labels from filtered feature rows")
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


def derive_execution_target_5d(frame: pd.DataFrame) -> pd.Series:
    """Use the T+1 open to T+5 close return when the feature file has it."""
    if "target_open_close_5d" in frame:
        return frame["target_open_close_5d"]
    # Older processed files cannot reconstruct the overnight gaps.  Fall back
    # explicitly to the close-to-close label rather than silently inventing a
    # different execution price.
    return derive_target_5d(frame)


def metric_dict(metric: Metric) -> Dict[str, float]:
    return asdict(metric)


def cross_section_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Normalize completed factor outputs for portfolio combination only."""
    return values.groupby(dates, sort=False).rank(pct=True) - 0.5


def evaluate_horizons(
    dates: pd.Series,
    score: pd.Series,
    targets: dict[int, pd.Series],
    masks: dict[int, dict[str, pd.Series]],
    instruments: pd.Series,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate the same horizon and split layout for candidates and the baseline."""
    return {
        str(horizon): {
            split: metric_dict(evaluate_score(dates, score, targets[horizon], masks[horizon][split], instruments))
            for split in ("train", "validation", "backtest")
        }
        for horizon in EVALUATION_HORIZONS
    }


def gate_failures(
    metric: dict[str, float],
    horizon: int,
    direction: int,
    *,
    ic_floor: float | None,
    icir_floor: float | None,
    hit_floor: float | None,
    min_coverage: float,
    max_turnover: float,
) -> list[str]:
    """Report failures under the strict Miner gate with optional threshold overrides."""
    thresholds = PAPER_HORIZON_THRESHOLDS[horizon]
    ic_floor = thresholds["ic"] if ic_floor is None else ic_floor
    icir_floor = thresholds["icir"] if icir_floor is None else icir_floor
    hit_floor = 0.5 + PAPER_MIN_HIT_DEVIATION if hit_floor is None else hit_floor
    directional_hit = (
        metric["rank_ic_hit_ratio"]
        if direction > 0
        else 1.0 - metric["rank_ic_hit_ratio"]
    )
    checks = {
        "insufficient_ic": direction * metric["mean_daily_ic"] > ic_floor,
        "insufficient_rank_ic": direction * metric["mean_daily_rank_ic"] > ic_floor,
        "insufficient_icir": direction * metric["icir"] > icir_floor,
        "insufficient_rank_icir": direction * metric["rank_icir"] > icir_floor,
        "unstable_daily_direction": directional_hit > hit_floor,
        "low_coverage": metric["coverage"] > min_coverage,
        "high_or_unknown_turnover": metric["turnover"] < max_turnover,
    }
    return [name for name, passed in checks.items() if not passed]
