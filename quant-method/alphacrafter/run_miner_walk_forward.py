#!/usr/bin/env python3
"""Select Miner-lite factors on past data and emit strictly forward signals.

For each evaluation year, directions and factor selection use only observations
through the preceding year.  The final 2024-2026 signal is therefore selected
using data available through 2023, without using the final holdout for factor
admission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from alphax.factor_dsl import Candidate, evaluate_candidate_score
from alphax.miner_candidates import BASELINE_FEATURES, template_candidates
from alphax.miner_metrics import cross_section_rank, evaluate_score
from alphax.miner_registry import candidate_from_factor
from temporal import META_COLUMNS, is_feature_column


def train_rank_ic(
    dates: pd.Series, target: pd.Series, score: pd.Series, train_end: str
) -> tuple[float, float]:
    """Use the shared Miner metric definitions for rolling selection."""
    metric = evaluate_score(dates, score, target, dates <= pd.Timestamp(train_end))
    return metric.mean_daily_rank_ic, metric.rank_icir


def select_factors(
    dates: pd.Series,
    target: pd.Series,
    features: pd.DataFrame,
    candidates: List[Candidate],
    train_end: str,
    min_abs_train_ic: float,
) -> List[Dict[str, object]]:
    selected = []
    for candidate in candidates:
        score = evaluate_candidate_score(candidate, features)
        mean_ic, icir = train_rank_ic(dates, target, score, train_end)
        if pd.isna(mean_ic) or abs(mean_ic) < min_abs_train_ic:
            continue
        direction = 1 if mean_ic >= 0 else -1
        selected.append(
            {
                "candidate_id": candidate.candidate_id,
                "op": candidate.op,
                "features": list(candidate.features),
                "expression": candidate.expression,
                "expression_spec": candidate.expression_spec,
                "direction": direction,
                "train_end": train_end,
                "train_rank_ic": mean_ic,
                "train_directional_rank_ic": direction * mean_ic,
                "train_rank_icir": icir,
            }
        )
    selected.sort(key=lambda item: item["train_directional_rank_ic"], reverse=True)
    return selected


def composite_score(features: pd.DataFrame, selected: Sequence[Dict[str, object]]) -> pd.Series:
    if not selected:
        raise ValueError("Cannot build a composite without selected factors")
    scores = []
    for factor in selected:
        candidate = candidate_from_factor(factor)
        raw = evaluate_candidate_score(candidate, features)
        scores.append(int(factor["direction"]) * cross_section_rank(raw, features["date"]))
    return pd.concat(scores, axis=1).sum(axis=1, min_count=len(scores)) / len(scores)


def validation_portfolio_metrics(
    dates: pd.Series,
    instruments: pd.Series,
    target: pd.Series,
    score: pd.Series,
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    cost_rate: float,
) -> Dict[str, float]:
    """Estimate net top-k performance for validation-only factor-count choice."""
    mask = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    frame = pd.DataFrame(
        {"date": dates, "instrument": instruments, "score": score, "target": target}
    ).loc[mask]
    daily_returns = []
    turnovers = []
    previous = set()
    for _, group in frame.groupby("date", sort=True):
        desired = set(group.nlargest(min(topk, len(group)), "score")["instrument"])
        if not previous:
            holdings = desired
            turnover = 0.95
        else:
            sell_candidates = group[group["instrument"].isin(previous - desired)].sort_values("score")
            sell = set(sell_candidates.head(min(n_drop, len(sell_candidates)))["instrument"])
            buy_candidates = group[group["instrument"].isin(desired - previous)].sort_values("score", ascending=False)
            buy = set(buy_candidates.head(len(sell))["instrument"])
            holdings = (previous - sell) | buy
            turnover = 2.0 * len(sell) / max(1, topk)
        gross = float(group[group["instrument"].isin(holdings)]["target"].mean())
        daily_returns.append(gross - cost_rate * turnover)
        turnovers.append(turnover)
        previous = holdings
    returns = pd.Series(daily_returns, dtype=float)
    cumulative = float((1.0 + returns).prod() - 1.0)
    volatility = float(returns.std(ddof=1) * (252.0 ** 0.5)) if len(returns) > 1 else float("nan")
    sharpe = float(returns.mean() / returns.std(ddof=1) * (252.0 ** 0.5)) if returns.std(ddof=1) else float("nan")
    return {
        "validation_cumulative_return": cumulative,
        "validation_sharpe": sharpe,
        "validation_annualized_volatility": volatility,
        "validation_mean_turnover": float(pd.Series(turnovers).mean()),
        "validation_days": int(len(returns)),
    }


def choose_factor_count(
    dates: pd.Series,
    instruments: pd.Series,
    target: pd.Series,
    features: pd.DataFrame,
    selected: List[Dict[str, object]],
    start: str,
    end: str,
    factor_counts: Sequence[int],
    topk: int,
    n_drop: int,
    cost_rate: float,
) -> tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    trials = []
    for count in factor_counts:
        if count > len(selected) or count < 1:
            continue
        metrics = validation_portfolio_metrics(
            dates,
            instruments,
            target,
            composite_score(features, selected[:count]),
            start,
            end,
            topk,
            n_drop,
            cost_rate,
        )
        trials.append({"factor_count": count, **metrics})
    if not trials:
        raise ValueError("No factor count can be evaluated")
    best = max(trials, key=lambda item: item["validation_cumulative_return"])
    return selected[: int(best["factor_count"])], trials


def build_signal(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    selected: List[Dict[str, object]],
    end: str,
) -> pd.DataFrame:
    if not selected:
        raise ValueError("No factors passed the train-only admission gate")
    composite = composite_score(features, selected)
    signal = pd.DataFrame(
        {
            "datetime": frame["date"].to_numpy(),
            "instrument": frame["ticker"].to_numpy(),
            "prediction": composite.to_numpy(dtype=float),
        }
    )
    signal = signal[signal["datetime"] <= pd.Timestamp(end)]
    return signal.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run nested, cost-aware Miner-lite walk-forward selection")
    parser.add_argument("--input", type=Path, default=base / "data" / "processed" / "sp500_features.csv.gz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_miner_walk_forward",
    )
    parser.add_argument("--max-factors", type=int, default=8)
    parser.add_argument("--factor-counts", default="1,2,4,8")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=10)
    parser.add_argument("--validation-cost-rate", type=float, default=0.0015)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--min-abs-train-ic", type=float, default=0.010)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, compression="infer", parse_dates=["date"])
    available = [column for column in frame.columns if is_feature_column(column) and column not in BASELINE_FEATURES]
    frame = frame.dropna(subset=available + ["target_open_close"]).reset_index(drop=True)
    dates = frame["date"]
    target = frame["target_open_close"]
    factor_counts = [int(value) for value in args.factor_counts.split(",") if value.strip()]
    factor_counts = [value for value in factor_counts if value <= args.max_factors]
    features = frame[["date", "ticker", *available]]
    candidates = template_candidates(available, args.max_pairs)

    windows = {
        # Label end dates exclude the last signal date before the next window;
        # its next-session outcome would not yet be known at the cutoff.
        "2021": {"train_end": "2019-12-30", "valid_start": "2020-01-02", "valid_end": "2020-12-30", "start": "2021-01-04", "end": "2021-12-31"},
        "2022": {"train_end": "2020-12-30", "valid_start": "2021-01-04", "valid_end": "2021-12-30", "start": "2022-01-03", "end": "2022-12-30"},
        "2023": {"train_end": "2021-12-30", "valid_start": "2022-01-03", "valid_end": "2022-12-29", "start": "2023-01-03", "end": "2023-12-29"},
        "2024_2026": {"train_end": "2022-12-29", "valid_start": "2023-01-03", "valid_end": "2023-12-28", "start": "2024-01-02", "end": "2026-02-26"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = {}
    for name, window in windows.items():
        ranked_factors = select_factors(
            dates,
            target,
            features,
            candidates,
            window["train_end"],
            args.min_abs_train_ic,
        )
        selected, validation_trials = choose_factor_count(
            dates,
            frame["ticker"],
            target,
            features,
            ranked_factors,
            window["valid_start"],
            window["valid_end"],
            factor_counts,
            args.topk,
            args.n_drop,
            args.validation_cost_rate,
        )
        signal = build_signal(frame, features, selected, window["end"])
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        signal.to_csv(output_dir / "predictions.csv", index=False)
        payload = {
            "window": window,
            "selection": {
                "factor_counts": factor_counts,
                "topk": args.topk,
                "n_drop": args.n_drop,
                "validation_cost_rate": args.validation_cost_rate,
                "min_abs_train_ic": args.min_abs_train_ic,
                "candidate_count": len(candidates),
                "train_eligible_factor_count": len(ranked_factors),
                "validation_trials": validation_trials,
                "selected_factors": selected,
            },
            "signal_file": "predictions.csv",
        }
        (output_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
        registry[name] = payload
        print(name, [factor["candidate_id"] for factor in selected])

    (args.output_dir / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
