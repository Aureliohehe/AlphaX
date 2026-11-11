#!/usr/bin/env python3
"""Walk-forward screening and combination of cross-sectional technical factors.

The factor direction and selected features for each evaluation year are learned
only from observations before that year.  This is deliberately simpler than
XGBoost: it provides an honest test of whether the other technical factors add
information beyond one-day reversal before changing the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from data_contract import META_COLUMNS, is_feature_column


EVAL_YEARS = (2021, 2022, 2023)
TOP_K = (1, 2, 4, 8, 12)
FEATURE_EXCLUDE = META_COLUMNS
# ``one_day_reversal`` is exactly ``-return_1d``.  Exclude both from the
# candidate pool so the screen measures incremental information beyond the
# simple reversal baseline rather than counting the same factor twice.
BASELINE_FEATURES = {"return_1d", "one_day_reversal"}


def daily_feature_ic(
    dates: pd.Series, ranked_features: pd.DataFrame, ranked_target: pd.Series, mask: pd.Series
) -> pd.DataFrame:
    """Return mean daily IC and ICIR for each feature under ``mask``."""
    rows: List[Dict[str, float]] = []
    selected_dates = dates[mask]
    for _, indices in selected_dates.groupby(selected_dates).groups.items():
        idx = np.asarray(indices)
        x = ranked_features.loc[idx].to_numpy(dtype=float)
        y = ranked_target.loc[idx].to_numpy(dtype=float)
        for column_index, feature in enumerate(ranked_features.columns):
            values = x[:, column_index]
            valid = np.isfinite(values) & np.isfinite(y)
            if valid.sum() < 3:
                continue
            xv = values[valid]
            yv = y[valid]
            xv = xv - xv.mean()
            yv = yv - yv.mean()
            denominator = np.sqrt(np.dot(xv, xv) * np.dot(yv, yv))
            if denominator > 0:
                rows.append({"feature": feature, "ic": float(np.dot(xv, yv) / denominator)})

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No daily factor ICs could be calculated")
    summary = result.groupby("feature")["ic"].agg(["mean", "std"])
    summary["rank_icir"] = summary["mean"] / summary["std"]
    summary["hit_ratio"] = result.groupby("feature")["ic"].apply(lambda values: float((values > 0).mean()))
    return summary.rename(columns={"mean": "rank_ic"}).reset_index()


def score_metrics(dates: pd.Series, score: pd.Series, target: pd.Series, mask: pd.Series) -> Dict[str, float]:
    """Evaluate a score by daily Rank IC and top/bottom decile spread."""
    frame = pd.DataFrame(
        {"date": dates, "score": score, "target": target}, index=dates.index
    ).loc[mask].dropna()
    daily_ic: List[float] = []
    spreads: List[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 3 or group["score"].nunique() < 2:
            continue
        daily_ic.append(float(group["score"].corr(group["target"], method="spearman")))
        count = max(1, len(group) // 10)
        ordered = group.sort_values("score")
        spreads.append(float(ordered["target"].iloc[-count:].mean() - ordered["target"].iloc[:count].mean()))
    ic = pd.Series(daily_ic).dropna()
    spread = pd.Series(spreads).dropna()
    return {
        "rows": int(len(frame)),
        "mean_daily_rank_ic": float(ic.mean()),
        "daily_rank_ic_std": float(ic.std()),
        "rank_icir": float(ic.mean() / ic.std()),
        "rank_ic_hit_ratio": float((ic > 0).mean()),
        "mean_top_bottom_decile_spread": float(spread.mean()),
    }


def factor_score(ranked_features: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    """Equal-weight selected factors after applying training-only directions."""
    score = pd.Series(0.0, index=ranked_features.index)
    for row in selected.itertuples(index=False):
        score = score + float(row.direction) * ranked_features[row.feature]
    return score / len(selected)


def main() -> None:
    parser_input = Path(__file__).resolve().parent / "data" / "processed" / "sp500_features.csv.gz"
    parser_output = Path(__file__).resolve().parent / "data" / "qlib" / "factor_optimization"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=parser_input)
    parser.add_argument("--output-dir", type=Path, default=parser_output)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, compression="infer", parse_dates=["date"])
    frame["one_day_reversal"] = -frame["return_1d"]
    all_features = [column for column in frame.columns if is_feature_column(column)]
    features = [column for column in all_features if column not in BASELINE_FEATURES]
    dates = frame["date"]
    target = frame["target_open_close"] if "target_open_close" in frame else frame["target_1d"]

    # Rank transformation is calculated date by date.  Ranking is also what
    # makes this screen comparable to the final cross-sectional IC objective.
    ranked_features = frame[features].groupby(dates, sort=False).rank(pct=True)
    ranked_baseline = frame["one_day_reversal"].groupby(dates, sort=False).rank(pct=True)
    ranked_target = target.groupby(dates, sort=False).rank(pct=True)

    all_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    detail: Dict[str, object] = {"evaluation_years": {}, "features": features}

    # Rolling screens: only dates before the evaluation year determine factor
    # direction and ranking.  The evaluation year is never used for selection.
    for year in EVAL_YEARS:
        train_mask = dates.dt.year < year
        eval_mask = dates.dt.year == year
        ic_table = daily_feature_ic(dates, ranked_features, ranked_target, train_mask)
        ic_table["abs_rank_ic"] = ic_table["rank_ic"].abs()
        ic_table = ic_table.sort_values("abs_rank_ic", ascending=False)

        year_detail = {"training_factor_ic": ic_table.to_dict(orient="records"), "models": {}}
        for top_k in TOP_K:
            selected = ic_table.head(top_k).copy()
            selected["direction"] = np.sign(selected["rank_ic"]).replace(0, 1)
            score = factor_score(ranked_features, selected)
            train_metrics = score_metrics(dates, score, target, train_mask)
            eval_metrics = score_metrics(dates, score, target, eval_mask)
            name = f"top_{top_k}"
            year_detail["models"][name] = {
                "selected": selected[["feature", "rank_ic", "direction"]].to_dict(orient="records"),
                "train": train_metrics,
                "evaluation": eval_metrics,
            }
            for split_name, metrics in (("train", train_metrics), (f"year_{year}", eval_metrics)):
                all_rows.append(
                    {
                        "evaluation_year": year,
                        "model": name,
                        "split": split_name,
                        **metrics,
                    }
                )
            if top_k == 4:
                selected_rows.extend(
                    {
                        "evaluation_year": year,
                        "feature": row.feature,
                        "training_rank_ic": row.rank_ic,
                        "direction": row.direction,
                    }
                    for row in selected.itertuples(index=False)
                )

        # Always include the simple reversal baseline for every rolling window.
        baseline_score = ranked_baseline
        for split_name, mask in (("train", train_mask), (f"year_{year}", eval_mask)):
            all_rows.append(
                {
                    "evaluation_year": year,
                    "model": "one_day_reversal",
                    "split": split_name,
                    **score_metrics(dates, baseline_score, target, mask),
                }
            )
        detail["evaluation_years"][str(year)] = year_detail

    # Final holdout-style report: select factors using 2016-2022 only, then
    # report 2023 and 2024-2026 separately.  2023 remains a model-selection
    # period in the broader project, so the 2024-2026 numbers are the cleaner
    # check for this factor screen.
    train_mask = dates.dt.year <= 2022
    valid_mask = dates.dt.year == 2023
    backtest_mask = dates.dt.year >= 2024
    ic_table = daily_feature_ic(dates, ranked_features, ranked_target, train_mask)
    ic_table["abs_rank_ic"] = ic_table["rank_ic"].abs()
    ic_table = ic_table.sort_values("abs_rank_ic", ascending=False)
    final_detail = {}
    for top_k in TOP_K:
        selected = ic_table.head(top_k).copy()
        selected["direction"] = np.sign(selected["rank_ic"]).replace(0, 1)
        score = factor_score(ranked_features, selected)
        final_detail[f"top_{top_k}"] = {
            "selected": selected[["feature", "rank_ic", "direction"]].to_dict(orient="records"),
            "validation_2023": score_metrics(dates, score, target, valid_mask),
            "backtest_2024_2026": score_metrics(dates, score, target, backtest_mask),
        }
    final_detail["one_day_reversal"] = {
        "validation_2023": score_metrics(dates, ranked_baseline, target, valid_mask),
        "backtest_2024_2026": score_metrics(dates, ranked_baseline, target, backtest_mask),
    }
    detail["final_holdout"] = final_detail
    detail["baseline_features"] = sorted(BASELINE_FEATURES)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output_dir / "rolling_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "rolling_top4_features.csv", index=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False, default=float) + "\n"
    )

    print("Final holdout-style comparison (directions learned through 2022):")
    for name, result in final_detail.items():
        valid = result["validation_2023"]
        backtest = result["backtest_2024_2026"]
        print(
            f"{name:18s} valid IC={valid['mean_daily_rank_ic']:.6f} "
            f"backtest IC={backtest['mean_daily_rank_ic']:.6f} "
            f"backtest ICIR={backtest['rank_icir']:.4f}"
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
