#!/usr/bin/env python3
"""Create a paper-style diagnostic figure for the frozen XGBoost baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "model": "#0072B2",
    "baseline": "#D55E00",
    "zero": "#666666",
    "train": "#F0F0F0",
    "valid": "#FFF3CD",
    "backtest": "#E8F1FB",
}


def daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for date, group in frame.groupby("datetime", sort=True):
        group = group.dropna(subset=["label", "prediction", "one_day_reversal"])
        if len(group) < 3:
            continue
        model_ic = group["label"].corr(group["prediction"], method="spearman")
        baseline_ic = group["label"].corr(group["one_day_reversal"], method="spearman")
        count = max(1, len(group) // 10)
        model_order = group.sort_values("prediction")
        baseline_order = group.sort_values("one_day_reversal")
        rows.append(
            {
                "datetime": date,
                "model_rank_ic": model_ic,
                "baseline_rank_ic": baseline_ic,
                "model_spread": model_order["label"].iloc[-count:].mean()
                - model_order["label"].iloc[:count].mean(),
                "baseline_spread": baseline_order["label"].iloc[-count:].mean()
                - baseline_order["label"].iloc[:count].mean(),
            }
        )
    result = pd.DataFrame(rows).set_index("datetime").sort_index()
    result["model_rolling_ic"] = result["model_rank_ic"].rolling(60, min_periods=30).mean()
    result["baseline_rolling_ic"] = result["baseline_rank_ic"].rolling(60, min_periods=30).mean()
    result["model_cumulative_spread"] = result["model_spread"].cumsum() * 100
    result["baseline_cumulative_spread"] = result["baseline_spread"].cumsum() * 100
    return result


def add_split_shading(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    valid_end = pd.Timestamp("2024-01-01")
    ax.axvspan(start, valid_end, color=COLORS["valid"], alpha=0.7, lw=0)
    ax.axvspan(valid_end, end, color=COLORS["backtest"], alpha=0.7, lw=0)
    ax.axvline(valid_end, color="#888888", ls="--", lw=0.9)


def main() -> None:
    base = Path(__file__).resolve().parent
    default_dir = base / "data" / "qlib" / "xgboost_baseline"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path, default=default_dir / "figures")
    parser.add_argument("--start-date", default="2023-01-03")
    args = parser.parse_args()

    predictions = pd.read_csv(
        args.input_dir / "predictions.csv", parse_dates=["datetime"]
    )
    predictions = predictions[predictions["datetime"] >= pd.Timestamp(args.start_date)].copy()
    if predictions.empty:
        raise SystemExit("No predictions remain after --start-date")
    daily = daily_metrics(predictions)
    importance = pd.read_csv(args.input_dir / "feature_importance.csv").nlargest(15, "gain")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1.15, 1.0, 1.35]}
    )
    start, end = daily.index.min(), daily.index.max()

    # Panel A: cumulative long-short decile spread.
    ax = axes[0]
    add_split_shading(ax, start, end)
    ax.plot(
        daily.index,
        daily["model_cumulative_spread"],
        color=COLORS["model"],
        lw=1.8,
        label="XGBoost",
    )
    ax.plot(
        daily.index,
        daily["baseline_cumulative_spread"],
        color=COLORS["baseline"],
        lw=1.5,
        label="One-day reversal",
    )
    ax.axhline(0, color=COLORS["zero"], lw=0.8)
    ax.set_title("(a) Cumulative top-bottom decile return spread")
    ax.set_ylabel("Cumulative spread (%)")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.text(pd.Timestamp("2023-06-01"), 0.94, "Validation", transform=ax.get_xaxis_transform(), color="#806000")
    ax.text(pd.Timestamp("2024-06-15"), 0.94, "Backtest", transform=ax.get_xaxis_transform(), color="#2F5597")

    # Panel B: rolling cross-sectional Rank IC.
    ax = axes[1]
    add_split_shading(ax, start, end)
    ax.plot(
        daily.index,
        daily["model_rolling_ic"],
        color=COLORS["model"],
        lw=1.5,
        label="XGBoost (60-day rolling)",
    )
    ax.plot(
        daily.index,
        daily["baseline_rolling_ic"],
        color=COLORS["baseline"],
        lw=1.3,
        label="Reversal (60-day rolling)",
    )
    ax.axhline(0, color=COLORS["zero"], lw=0.8)
    ax.set_title("(b) Rolling cross-sectional Rank IC")
    ax.set_ylabel("Rank IC")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    # Panel C: global tree gain importance.
    ax = axes[2]
    importance = importance.sort_values("gain")
    labels = importance["feature"].str.replace("_", " ", regex=False)
    ax.barh(labels, importance["gain"], color=COLORS["model"], alpha=0.85)
    ax.set_title("(c) XGBoost feature importance by average gain")
    ax.set_xlabel("Gain")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

    for ax in axes:
        ax.grid(axis="y", ls=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)

    fig.suptitle(
        "XGBoost baseline: out-of-sample ranking performance and feature importance",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.99,
        0.01,
        "Shaded: 2023 validation / 2024–2026 backtest",
        ha="right",
        color="#555555",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "xgb_baseline_paper_figure.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "xgb_baseline_paper_figure.pdf", bbox_inches="tight")
    daily.to_csv(args.output_dir / "daily_metrics.csv")
    print(f"Saved figure to {args.output_dir / 'xgb_baseline_paper_figure.png'}")


if __name__ == "__main__":
    main()
