#!/usr/bin/env python3
"""Fixed Alpha158-style XGBoost diagnostic on historical development folds only.

This is a separate model route, NOT Miner admission, a tradable strategy, or
an untouched holdout evaluation. Dates in each fold are signal dates; labels
must settle no later than that fold's own end. No validation labels enter fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract import calendar_audit, read_calendar, resolve_label_ends

FOLDS = (
    ("2021", "2020-12-31", "2021-01-04", "2021-12-31"),
    ("2022", "2021-12-31", "2022-01-03", "2022-12-30"),
    ("2023", "2022-12-30", "2023-01-03", "2023-12-29"),
)
TRAIN_START = "2016-01-04"
PARAMS = {
    "objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 3,
    "eta": 0.05, "min_child_weight": 20, "subsample": 0.8,
    "colsample_bytree": 0.8, "lambda": 10, "nthread": 4, "seed": 42,
    "tree_method": "hist",
}
ROUNDS = 25  # Fixed before evaluating the folds; no early stopping or grid search.


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def candidate_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a nonempty, frozen candidate list")
    features = []
    for item in payload:
        if not isinstance(item, dict) or len(item.get("features", [])) != 1:
            raise ValueError("Model diagnostic requires one source feature per candidate")
        feature = item["features"][0]
        if not isinstance(feature, str) or not feature.startswith("alpha158_") or item.get("expression") != {"op": "feature", "feature": feature}:
            raise ValueError("Only raw, single-column alpha158_ candidates are supported")
        features.append(feature)
    if len(set(features)) != len(features):
        raise ValueError("Duplicate candidate feature")
    return features


def fold_masks(frame: pd.DataFrame, train_end: str, valid_start: str, valid_end: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates, ends = frame["date"], frame["label_end_1d"]
    train = dates.between(TRAIN_START, train_end) & ends.notna() & (ends <= pd.Timestamp(train_end)) & frame["target_open_close"].notna()
    validation = dates.between(valid_start, valid_end)
    scored = validation & ends.notna() & (ends <= pd.Timestamp(valid_end)) & frame["target_open_close"].notna()
    if not train.any() or not scored.any():
        raise ValueError("No settled training/validation labels")
    if train.any() and (ends[train] > pd.Timestamp(train_end)).any():
        raise AssertionError("Training label crossed the fold boundary")
    return train, validation, scored


def daily_metrics(frame: pd.DataFrame, score: pd.Series, mask: pd.Series) -> tuple[dict, pd.DataFrame]:
    data = pd.DataFrame({"date": frame.loc[mask, "date"], "label": frame.loc[mask, "target_open_close"], "score": score.loc[mask]})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    daily = []
    for date, group in data.groupby("date", sort=True):
        if len(group) < 3 or group["label"].nunique() < 2 or group["score"].nunique() < 2:
            continue
        daily.append({"date": str(date.date()), "rows": len(group),
                      "ic": float(group["score"].corr(group["label"])),
                      "rank_ic": float(group["score"].corr(group["label"], method="spearman"))})
    results = pd.DataFrame(daily)
    if results.empty:
        raise ValueError("No valid cross-sectional IC days")
    ic, rank_ic = results["ic"], results["rank_ic"]
    return {
        "rows": int(len(data)), "ic_days": int(len(results)),
        "mean_daily_ic": float(ic.mean()),
        "icir": float(ic.mean() / ic.std(ddof=0)) if ic.std(ddof=0) > 0 else None,
        "mean_daily_rank_ic": float(rank_ic.mean()),
        "rank_icir": float(rank_ic.mean() / rank_ic.std(ddof=0)) if rank_ic.std(ddof=0) > 0 else None,
        "rank_ic_hit_ratio": float((rank_ic > 0).mean()),
    }, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.input, args.calendar, args.candidate_file = (path.resolve(strict=True) for path in (args.input, args.calendar, args.candidate_file))
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty; no overwrite/resume")
    features = candidate_features(args.candidate_file)
    columns = ["date", "ticker", "label_end_1d", "target_open_close", "return_1d", *features]
    # Read only the frozen feature subset; never rank on a future cross-section.
    frame = pd.read_csv(args.input, usecols=columns, parse_dates=["date", "label_end_1d"])
    frame = frame[frame["date"] <= pd.Timestamp(FOLDS[-1][3])].copy()
    if frame.empty or frame[["date", "ticker"]].isna().any().any() or frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Empty/invalid feature panel keys")
    reference = read_calendar(args.calendar)
    audit = calendar_audit(frame["date"], reference, frame["date"].min(), FOLDS[-1][3])
    if not audit["matches_reference"]:
        raise ValueError(f"Missing/nonmarket feature sessions: {audit['missing_sessions'][:10]}, {audit['unexpected_sessions'][:10]}")
    resolve_label_ends(frame, 1, require_explicit=True, calendar=reference)
    # Per-session ranks use only today's available features. They do not use
    # future returns or train-fitted statistics, and preserve missing values.
    ranked = frame[features].groupby(frame["date"], sort=False).rank(pct=True).astype("float32")
    frame["reversal"] = -frame["return_1d"].astype(float)
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise SystemExit("Use the tradingagents environment with XGBoost installed") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"type": "exploratory_model_not_miner_or_strategy", "features": features,
                "inputs": {name: fingerprint(path) for name, path in (("panel", args.input), ("calendar", args.calendar), ("candidates", args.candidate_file))},
                "code": fingerprint(Path(__file__)), "calendar_audit": audit, "model": "xgboost_hist_fixed",
                "params": PARAMS, "rounds": ROUNDS, "folds": FOLDS,
                "library_versions": {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
                "warning": "Validation 2021-2023 is exploratory; 2023/2024+ have been viewed in prior experiments; no holdout or trading return"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries = []
    for name, train_end, valid_start, valid_end in FOLDS:
        train, validation, scored = fold_masks(frame, train_end, valid_start, valid_end)
        # Supervised labels are available ONLY after the exact label end; the
        # next fold refits rather than inheriting a future-trained model.
        y_train = frame.loc[train, "target_open_close"].groupby(frame.loc[train, "date"]).rank(pct=True).to_numpy(dtype="float32") - 0.5
        fit = xgb.DMatrix(ranked.loc[train, features].to_numpy(), label=y_train, feature_names=features)
        model = xgb.train(PARAMS, fit, num_boost_round=ROUNDS, verbose_eval=False)
        predictions = pd.Series(np.nan, index=frame.index)
        predictions.loc[validation] = model.predict(xgb.DMatrix(ranked.loc[validation, features].to_numpy(), feature_names=features))
        pred_frame = frame.loc[validation, ["date", "ticker"]].copy()
        pred_frame["prediction"] = predictions.loc[validation]
        pred_frame.to_csv(args.output_dir / f"predictions_{name}.csv", index=False)
        model.save_model(str(args.output_dir / f"model_{name}.json"))
        metrics, daily = daily_metrics(frame, predictions, scored)
        baseline, baseline_daily = daily_metrics(frame, frame["reversal"], scored)
        daily.rename(columns={"ic": "model_ic", "rank_ic": "model_rank_ic"}).merge(
            baseline_daily.rename(columns={"ic": "reversal_ic", "rank_ic": "reversal_rank_ic"}),
            on=["date", "rows"], how="outer", validate="one_to_one",
        ).to_csv(args.output_dir / f"daily_ic_{name}.csv", index=False)
        summaries.append({"fold": name, "train_end": train_end, "valid_start": valid_start, "valid_end": valid_end,
                          "train_rows": int(train.sum()), "validation_signal_rows": int(validation.sum()),
                          "validation_settled_rows": int(scored.sum()), "model": metrics, "reversal": baseline})
    for name, source in (("panel", args.input), ("calendar", args.calendar), ("candidates", args.candidate_file)):
        if fingerprint(source) != manifest["inputs"][name]:
            raise ValueError(f"Input changed during model run: {name}")
    if fingerprint(Path(__file__)) != manifest["code"]:
        raise ValueError("Model code changed during run")
    result = {"status": "completed", "folds": summaries, "limitations": manifest["warning"]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    print(json.dumps({"status": "completed", "folds": [{"fold": row["fold"], "model": row["model"], "reversal": row["reversal"]} for row in summaries]}, indent=2))


if __name__ == "__main__":
    main()
