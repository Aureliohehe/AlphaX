#!/usr/bin/env python3
"""Retrospective Qlib three-way ablation: Alpha158(20), 2 LLM hypotheses, combined.

NOT fresh OOS: LLM candidates were proposed after the historical dates and
2022-2023 have already been examined. The Yahoo-derived Qlib provider uses
current S&P constituents, not a certified point-in-time universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BACKTEST = HERE / "qlib_portfolio_backtest.py"
DEFAULT_PROVIDER = HERE / "data/qlib/sp500_provider"
BASE_NAMES = ("KMID2", "KUP2", "KLOW2", "KSFT2", "ROC5", "ROC10", "ROC20", "ROC30",
              "MA10", "MA30", "STD10", "STD30", "RSV20", "CORR20", "CNTD20", "SUMD20",
              "MAX20", "MIN20", "QTLU20", "QTLD20")
# AST equivalents of frozen roc30_ts_slope_30 and corr20_ts_mean_20;
# these two were found by post-hoc exploration, not PIT discovered in 2018.
CUSTOM = {"LLM_ROC30_SLOPE30": "Slope(Ref($close, 30)/$close, 30)",
          "LLM_CORR20_MEAN20": "Mean(Corr($close, Log($volume+1), 20), 20)"}
SETS = {"alpha158_20": list(BASE_NAMES), "llm_two": list(CUSTOM),
        "combined": list(BASE_NAMES) + list(CUSTOM)}
TRAIN = ("2018-01-02", "2020-12-31")
VALID = ("2021-01-04", "2021-12-31")
TEST = ("2022-01-03", "2023-12-29")
LABEL = "exact next-market-session $close/$open - 1 (Qlib provider quotes)"


def sha(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def purge(frame: pd.DataFrame, calendar: pd.DatetimeIndex, window: tuple[str, str]) -> pd.DataFrame:
    """Never train/validate on a label whose T+1 execution crosses a split."""
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    days = frame.index.get_level_values("datetime")
    next_dates = pd.Series(days.map(next_day), index=frame.index)
    return frame[(days >= pd.Timestamp(window[0])) & (days <= pd.Timestamp(window[1])) &
                 next_dates.le(pd.Timestamp(window[1])).to_numpy() & next_dates.notna().to_numpy()]


def prepare(frame: pd.DataFrame, names: list[str], calendar: pd.DatetimeIndex) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """Model preprocessing fitted on training only; no label-based stock filter at inference."""
    train = purge(frame, calendar, TRAIN)
    valid = purge(frame, calendar, VALID)
    if train.empty or valid.empty:
        raise ValueError("Insufficient train/validation history")
    train = train[np.isfinite(train["LABEL"])].copy()
    valid = valid[np.isfinite(valid["LABEL"])].copy()
    # Cross-sectional ranks are computed using visible same-day features only.
    ranked = frame[names].replace([np.inf, -np.inf], np.nan).groupby(level="datetime").rank(pct=True)
    train_x, valid_x = ranked.loc[train.index], ranked.loc[valid.index]
    return (train_x, train["LABEL"]), (valid_x, valid["LABEL"]), ranked


def features_from_qlib(provider: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    import qlib
    from qlib.constant import REG_US
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D
    qlib.init(provider_uri=str(provider), region=REG_US, expression_cache=None, dataset_cache=None)
    expressions, names = Alpha158.__new__(Alpha158).get_feature_config()
    lookup = dict(zip(names, expressions))
    if set(BASE_NAMES) - set(lookup):
        raise ValueError("Qlib Alpha158 configuration lacks selected official features")
    spec = {name: lookup[name] for name in BASE_NAMES}
    spec.update(CUSTOM)
    # A few months of warm-up for 30+30-day nested operators. Fetch quotes
    # from Qlib and construct next-session labels only where a stock has an
    # actual quote on the market's very next session (never its next late quote).
    start, end = "2017-06-01", TEST[1]
    instruments = D.list_instruments(D.instruments("all"), start_time=start, end_time=end, as_list=True)
    if not instruments:
        raise ValueError("Empty Qlib instrument universe")
    values = D.features(instruments, list(spec.values()) + ["$open", "$close"], start_time=start, end_time=end, freq="day")
    values.columns = list(spec) + ["QUOTE_OPEN", "QUOTE_CLOSE"]
    if values.empty or not values.index.is_unique:
        raise ValueError("Qlib returned empty or duplicate market panel")
    values = values.sort_index(level=["datetime", "instrument"]).replace([np.inf, -np.inf], np.nan)
    cal = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
    next_day = dict(zip(cal[:-1], cal[1:]))
    dates = pd.Series(values.index.get_level_values("datetime"), index=values.index)
    following_quote_date = dates.groupby(level="instrument").shift(-1)
    open_next = values["QUOTE_OPEN"].groupby(level="instrument").shift(-1)
    close_next = values["QUOTE_CLOSE"].groupby(level="instrument").shift(-1)
    label = (close_next / open_next - 1).where(following_quote_date.eq(dates.map(next_day)) & open_next.gt(0))
    values = values.drop(columns=["QUOTE_OPEN", "QUOTE_CLOSE"])
    values["LABEL"] = label
    return values, cal, spec


def train_predict(frame: pd.DataFrame, calendar: pd.DatetimeIndex, names: list[str], seed: int) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb
    (train_x, train_y), (valid_x, valid_y), ranked = prepare(frame, names, calendar)
    model = lgb.LGBMRegressor(n_estimators=60, learning_rate=.05, num_leaves=15,
                               max_depth=5, min_child_samples=150, n_jobs=4,
                               subsample=1., colsample_bytree=1., random_state=seed,
                               verbosity=-1)
    model.fit(train_x, train_y)
    check = model.predict(valid_x)
    valid_ric = pd.Series(check, index=valid_x.index).to_frame("score").join(valid_y).groupby(level="datetime").apply(
        lambda day: day["score"].corr(day["LABEL"], method="spearman"), include_groups=False).dropna()
    # Inference includes prior valid day for Qlib's T+1-open first test trade.
    dates = ranked.index.get_level_values("datetime")
    wanted = ranked.loc[(dates >= pd.Timestamp("2021-12-20")) & (dates <= pd.Timestamp(TEST[1]))]
    # Do not select or filter inference stocks by their forward LABEL.
    signal = pd.DataFrame({"datetime": wanted.index.get_level_values("datetime"),
                           "instrument": wanted.index.get_level_values("instrument"),
                           "prediction": model.predict(wanted)}).replace([np.inf, -np.inf], np.nan)
    signal = signal.dropna(subset=["prediction"]).sort_values(["datetime", "instrument"])
    if signal.empty:
        raise ValueError("No signal after model prediction")
    info = {"features": names, "train_rows": len(train_x), "valid_rows": len(valid_x),
            "valid_rank_ic_mean": float(valid_ric.mean()), "valid_rank_ic_days": len(valid_ric),
            "test_labels_not_used_for_training": True}
    return signal, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-dir", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    provider = args.provider_dir.resolve(strict=True)
    if not (provider / "calendars/day.txt").is_file() or not (provider / "instruments/all.txt").is_file():
        parser.error("Point --provider-dir to a populated Qlib day provider")
    import qlib
    manifest = {"scope": "retrospective_not_pit_or_fresh_oos", "provider": str(provider),
                "provider_calendar": sha(provider / "calendars/day.txt"),
                "provider_universe": sha(provider / "instruments/all.txt"),
                "script_sha": sha(Path(__file__)), "backtest_script_sha": sha(BACKTEST),
                "python": sys.version, "qlib": qlib.__version__, "train": TRAIN, "valid": VALID, "test": TEST,
                "label": LABEL, "seed": args.seed, "feature_sets": SETS,
                "config": {"topk": 50, "n_drop": 10, "risk_degree": .95, "open_cost": .0005,
                           "close_cost": .0005, "impact_cost": .0005, "deal_price": "open"},
                "warning": "US provider from current Yahoo constituent snapshot; custom factors proposed after test; test previously viewed. Do not call results new OOS alpha."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        frame, calendar, spec = features_from_qlib(provider)
        (args.output_dir / "factor_expressions.json").write_text(json.dumps(spec, indent=2) + "\n")
        results = {}
        for name, features in SETS.items():
            directory = args.output_dir / name
            directory.mkdir()
            predictions, info = train_predict(frame, calendar, features, args.seed)
            pred_file = directory / "predictions.csv"
            predictions.to_csv(pred_file, index=False)
            cmd = [sys.executable, str(BACKTEST), "--predictions", str(pred_file),
                   "--provider-dir", str(provider), "--output-dir", str(directory / "backtest"),
                   "--start", TEST[0], "--end", TEST[1], "--topk", "50", "--n-drop", "10",
                   "--risk-degree", ".95", "--open-cost", ".0005", "--close-cost", ".0005", "--impact-cost", ".0005"]
            with (directory / "backtest.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((directory / "backtest/summary.json").read_text())
            results[name] = {"train_valid": info, "portfolio": metrics}
            (args.output_dir / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        if len({(r["portfolio"]["start"], r["portfolio"]["end"], r["portfolio"]["trading_days"],
                 r["portfolio"]["benchmark_cumulative_return"]) for r in results.values()}) != 1:
            raise ValueError("Qlib backtest windows/benchmarks differ across configurations")
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        lines = ["# Qlib three-way retrospective ablation", "", manifest["warning"], "",
                 "Same US day provider, same training/validation/test dates, next-day open executions, Topk50 Drop10 95% exposure, buy/sell 5bp and impact 5bp.",
                 "The 2 custom LLM ASTs were translated to Qlib expressions, not independently discovered in historical time. Model is fixed LightGBM, not QuantaAlpha's exact model.",
                 "", "| factors | valid Rank IC (2021) | 2022-23 net cumulative | benchmark | excess | Sharpe | MDD |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for key, row in results.items():
            m = row["portfolio"]
            lines.append(f"| {key} | {row['train_valid']['valid_rank_ic_mean']:.4f} | {m['cumulative_return']:.2%} | {m['benchmark_cumulative_return']:.2%} | {m['excess_cumulative_return']:.2%} | {m['sharpe']:.3f} | {m['max_drawdown']:.2%} |")
        (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
        (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_research"}) + "\n")
        print((args.output_dir / "report.md").read_text())
    except Exception as exc:
        (args.output_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
