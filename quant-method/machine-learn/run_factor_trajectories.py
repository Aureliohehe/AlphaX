#!/usr/bin/env python3
"""Budgeted, auditable factor trajectory pilot on a Qlib provider.

Deterministic mutations of TWO previously LLM-proposed hypotheses; no LLM API
call is made. No post-2021 data is read; 2021 has previously been examined, so
validation here is research feedback, not a fresh out-of-sample claim.
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

from run_qlib_three_way import BASE_NAMES, TRAIN, VALID, prepare, sha

HERE = Path(__file__).resolve().parent
BACKTEST = HERE / "qlib_portfolio_backtest.py"
PARENT_EXPR = {
    "roc30_slope30": "Slope(Ref($close, 30)/$close, 30)",
    "corr20_mean20": "Mean(Corr($close, Log($volume+1), 20), 20)",
}
PARENT_HYPOTHESIS = {
    "roc30_slope30": "Slope of the 30-session close ratio may reflect trend exhaustion",
    "corr20_mean20": "Persistence of price-volume correlation may proxy trend participation",
}
# Use only parameterized safe Qlib expressions. Never evaluate LLM-supplied Python.
TEMPLATES = {
    "roc30_slope30": lambda w: f"Slope(Ref($close, 30)/$close, {w})",
    "corr20_mean20": lambda w: f"Mean(Corr($close, Log($volume+1), 20), {w})",
}
DEFAULT_MUTATIONS = [
    {"parent_id": "roc30_slope30", "window": 20, "hypothesis": "Faster trend-slope window may capture earlier reversals"},
    {"parent_id": "roc30_slope30", "window": 45, "hypothesis": "Slower trend-slope window may filter noise"},
    {"parent_id": "corr20_mean20", "window": 10, "hypothesis": "Shorter smoothing may capture recent participation shifts"},
    {"parent_id": "corr20_mean20", "window": 30, "hypothesis": "Longer smoothing may reduce price-volume noise"},
]
EVAL_BUDGET = 8  # baseline + two parents + four mutations + one crossover


def build_trajectories(mutations: list[dict]) -> list[dict]:
    if not isinstance(mutations, list) or len(mutations) != 4:
        raise ValueError("Exactly four precommitted mutations required for the 8-evaluation budget")
    result = [{"id": "baseline_20", "phase": "baseline", "parent_ids": [],
               "hypothesis": "Qlib official Alpha158 20-feature benchmark", "expression": None}]
    for pid, expression in PARENT_EXPR.items():
        result.append({"id": pid, "phase": "parent_from_previous_llm", "parent_ids": [],
                       "hypothesis": PARENT_HYPOTHESIS[pid], "expression": expression})
    seen = set(PARENT_EXPR.values())
    for item in mutations:
        if not isinstance(item, dict) or set(item) != {"parent_id", "window", "hypothesis"}:
            raise ValueError("Mutations must contain parent_id, window and hypothesis only")
        pid, window = item["parent_id"], item["window"]
        if pid not in TEMPLATES or isinstance(window, bool) or not isinstance(window, int) or not 5 <= window <= 60:
            raise ValueError("Unsafe or unavailable parameterized mutation")
        hypothesis = item["hypothesis"]
        if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 300:
            raise ValueError("Mutation needs a short non-empty mechanism hypothesis")
        expr = TEMPLATES[pid](window)
        if expr in seen:
            raise ValueError("Duplicate parent/mutation expression")
        seen.add(expr)
        result.append({"id": f"{pid}_w{window}", "phase": "mutation", "parent_ids": [pid],
                       "hypothesis": hypothesis, "expression": expr})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate trajectory IDs")
    result.append({"id": "cross_short", "phase": "rank_crossover", "parent_ids": ["roc30_slope30_w20", "corr20_mean20_w10"],
                   "hypothesis": "Orthogonal price-trend and participation changes may jointly improve predictions",
                   "expression": "0.5*CSRank(roc30_slope30_w20)+0.5*CSRank(corr20_mean20_w10)"})
    if len(result) != EVAL_BUDGET:
        raise ValueError("Evaluation budget mismatch")
    return result


def load_data(provider: Path, trajectories: list[dict]) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    import qlib
    from qlib.constant import REG_US
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D
    qlib.init(provider_uri=str(provider), region=REG_US, expression_cache=None, dataset_cache=None)
    expressions, names = Alpha158.__new__(Alpha158).get_feature_config()
    official = dict(zip(names, expressions))
    spec = {name: official[name] for name in BASE_NAMES}
    spec.update({t["id"]: t["expression"] for t in trajectories if t["phase"] in ("parent_from_previous_llm", "mutation")})
    start, end = "2017-06-01", VALID[1]
    instruments = D.list_instruments(D.instruments("all"), start_time=start, end_time=end, as_list=True)
    if not instruments:
        raise ValueError("Empty Qlib provider universe")
    panel = D.features(instruments, list(spec.values()) + ["$open", "$close"],
                       start_time=start, end_time=end, freq="day")
    panel.columns = list(spec) + ["QUOTE_OPEN", "QUOTE_CLOSE"]
    if panel.empty or not panel.index.is_unique:
        raise ValueError("Empty/duplicate Qlib market panel")
    panel = panel.sort_index(level=["datetime", "instrument"]).replace([np.inf, -np.inf], np.nan)
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
    dates = pd.Series(panel.index.get_level_values("datetime"), index=panel.index)
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    following_date = dates.groupby(level="instrument").shift(-1)
    following_open = panel["QUOTE_OPEN"].groupby(level="instrument").shift(-1)
    following_close = panel["QUOTE_CLOSE"].groupby(level="instrument").shift(-1)
    panel["LABEL"] = (following_close / following_open - 1).where(following_date.eq(dates.map(next_day)) & following_open.gt(0))
    panel = panel.drop(columns=["QUOTE_OPEN", "QUOTE_CLOSE"])
    for t in trajectories:
        if t["phase"] == "rank_crossover":
            left, right = t["parent_ids"]
            panel[t["id"]] = (panel[left].groupby(level="datetime").rank(pct=True) +
                              panel[right].groupby(level="datetime").rank(pct=True)) / 2
            spec[t["id"]] = t["expression"]
    return panel, calendar, spec


def feature_redundancy(frame: pd.DataFrame, candidate: str, parents: list[str]) -> dict[str, float]:
    """Mean 2021 cross-sectional rank correlation (same-day features, no labels)."""
    result = {}
    dates = frame.index.get_level_values("datetime")
    visible = frame.loc[(dates >= pd.Timestamp(VALID[0])) & (dates <= pd.Timestamp(VALID[1]))]
    for parent in parents:
        panel = visible[[candidate, parent]].replace([np.inf, -np.inf], np.nan)
        daily = panel.groupby(level="datetime").apply(
            lambda day: day[candidate].corr(day[parent], method="spearman")).dropna()
        result[parent] = float(daily.mean()) if not daily.empty else float("nan")
    return result


def fit_validation(frame: pd.DataFrame, calendar: pd.DatetimeIndex, features: list[str], seed: int) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb
    (train_x, train_y), (valid_x, valid_y), ranked = prepare(frame, features, calendar)
    model = lgb.LGBMRegressor(n_estimators=60, learning_rate=.05, num_leaves=15, max_depth=5,
                              min_child_samples=150, n_jobs=4, subsample=1., colsample_bytree=1.,
                              random_state=seed, verbosity=-1)
    model.fit(train_x, train_y)
    scores = pd.Series(model.predict(valid_x), index=valid_x.index, name="score")
    joined = scores.to_frame().join(valid_y)
    daily_ric = joined.groupby(level="datetime").apply(
        lambda day: day["score"].corr(day["LABEL"], method="spearman"), include_groups=False).dropna()
    if daily_ric.empty:
        raise ValueError("No mature validation labels")
    dates = ranked.index.get_level_values("datetime")
    infer = ranked.loc[(dates >= pd.Timestamp("2020-12-20")) & (dates <= pd.Timestamp(VALID[1]))]
    predictions = pd.DataFrame({"datetime": infer.index.get_level_values("datetime"),
                                "instrument": infer.index.get_level_values("instrument"),
                                "prediction": model.predict(infer)}).replace([np.inf, -np.inf], np.nan)
    predictions = predictions.dropna(subset=["prediction"]).sort_values(["datetime", "instrument"])
    return predictions, {"train_rows": len(train_x), "validation_rows": len(valid_x),
                         "validation_rank_ic": float(daily_ric.mean()),
                         "validation_rank_icir": float(daily_ric.mean() / daily_ric.std(ddof=0)) if daily_ric.std(ddof=0) else None,
                         "validation_rank_ic_hit": float((daily_ric > 0).mean()), "validation_days": len(daily_ric)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-dir", type=Path, default=HERE / "data/qlib/sp500_provider")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mutations", type=Path, help="Optional precommitted JSON: exactly 4 safe parameterized proposals; no arbitrary code")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    provider = args.provider_dir.resolve(strict=True)
    if not (provider / "calendars/day.txt").exists() or not (provider / "instruments/all.txt").exists():
        parser.error("Provider needs calendars/day.txt and instruments/all.txt")
    if args.mutations:
        args.mutations = args.mutations.resolve(strict=True)
    proposals = json.loads(args.mutations.read_text()) if args.mutations else DEFAULT_MUTATIONS
    trajectories = build_trajectories(proposals)
    hashes = {str(path): sha(path) for path in (Path(__file__), BACKTEST, provider / "calendars/day.txt",
             provider / "instruments/all.txt", *([args.mutations] if args.mutations else []))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"scope": "retrospective_validation_feedback_only_no_test_or_live_library",
                "proposal_generator": "user_supplied_safe_json" if args.mutations else "deterministic_templates_not_new_llm_api",
                "evaluation_budget": EVAL_BUDGET, "train": TRAIN, "validation": VALID,
                "qlib_provider": str(provider), "seed": args.seed,
                "config": {"topk": 50, "n_drop": 10, "risk_degree": .95, "open_cost": .0005,
                           "close_cost": .0005, "impact_cost": .0005, "deal_price": "open"},
                "sources": hashes, "limitations": "Current Yahoo constituent snapshot; LLM parent proposals postdate 2021; validation already examined; no 2022+ scores consulted"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "trajectories_planned.json").write_text(json.dumps(trajectories, indent=2) + "\n")
    try:
        frame, calendar, specs = load_data(provider, trajectories)
        (args.output_dir / "expressions.json").write_text(json.dumps(specs, indent=2) + "\n")
        results = []
        for task in trajectories:
            features = list(BASE_NAMES) if task["phase"] == "baseline" else [*BASE_NAMES, task["id"]]
            predictions, metrics = fit_validation(frame, calendar, features, args.seed)
            directory = args.output_dir / task["id"]
            directory.mkdir()
            file = directory / "predictions.csv"
            predictions.to_csv(file, index=False)
            command = [sys.executable, str(BACKTEST), "--provider-dir", str(provider),
                       "--predictions", str(file), "--output-dir", str(directory / "backtest"),
                       "--start", VALID[0], "--end", VALID[1], "--topk", "50", "--n-drop", "10",
                       "--risk-degree", ".95", "--open-cost", ".0005", "--close-cost", ".0005", "--impact-cost", ".0005"]
            with (directory / "backtest.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            portfolio = json.loads((directory / "backtest/summary.json").read_text())
            related = (task["parent_ids"] if task["parent_ids"] else [] )
            row = {**task, "features": features, "metrics": metrics, "portfolio": portfolio,
                   "feature_similarity_to_parents": feature_redundancy(frame, task["id"], related) if related else {},
                   "status": "evaluated_retrospective_validation"}
            results.append(row)
            (args.output_dir / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        if len(results) != EVAL_BUDGET:
            raise ValueError("Budget incomplete; do not report partial experiment")
        if len({(r["portfolio"]["start"], r["portfolio"]["end"], r["portfolio"]["trading_days"],
                 r["portfolio"]["benchmark_cumulative_return"]) for r in results}) != 1:
            raise ValueError("Comparison windows/benchmarks differ")
        baseline = results[0]
        for r in results:
            r["feedback"] = {"delta_rank_ic_vs_base": r["metrics"]["validation_rank_ic"] - baseline["metrics"]["validation_rank_ic"],
                             "delta_portfolio_return_vs_base": r["portfolio"]["cumulative_return"] - baseline["portfolio"]["cumulative_return"],
                             "warning": "Same previously viewed 2021 validation used as feedback; not an independent test"}
        (args.output_dir / "trajectories.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        report = ["# Qlib fixed-budget factor trajectory pilot", "",
                  "RETROSPECTIVE 2021 validation feedback only; no 2022+ prices/labels accessed by this runner. Earlier LLM proposals postdate 2021, and the current-member Yahoo universe is not PIT.",
                  "8 evaluations (baseline + 2 parent additions + 4 parameterized mutations + 1 rank crossover), fixed LightGBM model and identical Qlib Topk50 Drop10 cost model. No new LLM API was called.",
                  "Each result is a model using the 20 Qlib baseline features PLUS the candidate; Rank IC is NOT an isolated factor IC.",
                  "Portfolio outcomes and Rank IC use the SAME 2021 validation window; never choose a threshold from these and call 2021 independent test.", "",
                  "| id | phase | parents | validation Rank IC | delta vs base | validation net cumulative | MDD |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for r in results:
            report.append(f"| {r['id']} | {r['phase']} | {', '.join(r['parent_ids']) or '-'} | {r['metrics']['validation_rank_ic']:.5f} | {r['feedback']['delta_rank_ic_vs_base']:.5f} | {r['portfolio']['cumulative_return']:.2%} | {r['portfolio']['max_drawdown']:.2%} |")
        report += ["", "Full hypotheses, expression lineage, rank correlations, costs and feedback in `trajectories.json`; Qlib daily portfolio books in `<id>/backtest/`.",
                   "No strict factor library or PIT snapshot was generated. `next_round_prompt.md` can be supplied to an LLM later, but the next proposal file is always bounded and schema-validated."]
        (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
        ranked = sorted(results[1:], key=lambda r: r["metrics"]["validation_rank_ic"], reverse=True)
        prompt = ["# Historical research feedback for FUTURE proposals only", "",
                  "This data was already viewed; no new historical PIT claim. Do not include 2022+ results. Evaluate only from these fixed 2021 validation metrics.",
                  "Propose exactly 4 new candidate JSON objects: parent_id, window (integer 5..60), hypothesis (max 300 chars).",
                  "Parents: roc30_slope30 (change Slope window); corr20_mean20 (change Mean smoothing window).",
                  "Avoid duplicate expressions. Explain why changes may improve cross-sectional predictability; do not promise a positive outcome.",
                  "Return a JSON list only. New file will be checked by a strict 8-run budget before execution.", "",
                  "Historical baseline Rank IC: " + str(baseline["metrics"]["validation_rank_ic"])]
        for r in ranked:
            prompt.append(f"- {r['id']} parents={r['parent_ids']} valid_2021_rank_ic={r['metrics']['validation_rank_ic']:.5f} "
                          f"delta_vs_base={r['feedback']['delta_rank_ic_vs_base']:.5f} "
                          f"portfolio_net={r['portfolio']['cumulative_return']:.2%} hypothesis={r['hypothesis']}")
        (args.output_dir / "next_round_prompt.md").write_text("\n".join(prompt) + "\n")
        for path, fingerprint in hashes.items():
            if sha(Path(path)) != fingerprint:
                raise ValueError(f"Source changed during trajectory run: {path}")
        (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_research"}) + "\n")
        print(json.dumps({"status": "completed_research", "evaluations": len(results),
                          "validation_baseline_rank_ic": baseline["metrics"]["validation_rank_ic"],
                          "best_candidate_id_excluding_baseline": ranked[0]["id"],
                          "best_candidate_rank_ic": ranked[0]["metrics"]["validation_rank_ic"],
                          "any_candidate_improved_rank_ic": any(r["feedback"]["delta_rank_ic_vs_base"] > 0 for r in results[1:])}, indent=2))
    except Exception as exc:
        (args.output_dir / "status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
