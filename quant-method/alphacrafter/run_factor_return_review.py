"""Audit the CSI300 return labels and run a frozen retrospective ranking comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from alphax.experiment import CSI300_DATA_FILES
from cn_portfolio import CNConfig, stamp_duty
from factor_return_model import (
    DATA, END, ENTRY, ENTRY_OPEN, EXIT, FEATURES, MIN_COVERAGE, MODEL_PARAMS, POLICY,
    REVIEW, START, TARGET, TOP_K, TRAIN_END, WINDOW_MONTHS, WeeklyEvaluation,
    features, performance, read_calendar, read_candidates, save_json, training_origins,
)
from run_cn_portfolio_research import load_quotes
from trader_lite import file_fingerprint

FAMILIES = {
    "momentum_and_trend": (
        "mf_momentum_60_skip5", "mf_momentum_120_skip20", "mf_trend_20_60", "mf_trend_60_120",
        "mf_risk_adjusted_momentum_60", "mf_volume_confirmed_trend", "mf_trend_consistency_60",
    ),
    "defensive_risk": ("mf_low_volatility_60", "mf_low_downside_risk_60"),
    "drawdown_and_reversal": ("mf_drawdown_60", "mf_reversal_20"),
    "volume_stability": ("mf_volume_stability_60",),
}
HELD_OUT = ("defensive_risk", "drawdown_and_reversal")
ADDITIONAL_FEATURES = (
    "factor_previous_12m", "factor_latest_12m", "factor_latest_6m", "index_previous_12m",
    "index_latest_12m", "index_latest_6m", "index_latest_12m_week_volatility",
    "latest_12m_mean_week_excess",
)
METHODS = ("history", "xgboost_v1", "xgboost_extended")
AUDIT_DATES = ("2012-07-01", "2015-07-01", "2019-07-01", "2024-07-01")


def candidate_family(candidate):
    field_families = {field: family for family, fields in FAMILIES.items() for field in fields}
    families = {field_families.get(name) for name in candidate.features}
    if None in families or len(families) != 1:
        raise ValueError(f"{candidate.candidate_id} does not belong to one frozen candidate family")
    return families.pop()


def coverage(weeks):
    count = len(weeks)
    usable = weeks.net.notna()
    member_failure = weeks.member_rows.lt(TOP_K)
    input_failure = ~member_failure & weeks.complete_inputs.lt(TOP_K)
    expression_failure = ~member_failure & ~input_failure & weeks.finite_scores.lt(TOP_K)
    missing_exit = weeks.finite_scores.ge(TOP_K) & weeks.missing_exit_sleeves.gt(0)
    # An AST can occasionally yield a finite value even if a raw field is missing.
    # Group failures by score availability first so buckets never double count.
    no_scores = weeks.finite_scores.lt(TOP_K)
    reasons = {
        "too_few_members_weeks": int((no_scores & member_failure).sum()),
        "too_few_finite_input_weeks": int((no_scores & input_failure).sum()),
        "expression_warmup_or_undefined_weeks": int(expression_failure.sum()),
        "missing_exit_weeks": int(missing_exit.sum()),
    }
    if int(usable.sum()) + sum(reasons.values()) != count:
        raise ValueError("Unclassified or overlapping weekly coverage loss")
    observed = float(usable.mean()) if count else 0.
    return {
        "scheduled_weeks": count, "observed_weeks": int(usable.sum()), "coverage": observed,
        "eligible": observed >= MIN_COVERAGE, **reasons,
        "weeks_with_missing_entry": int(weeks.missing_entry_sleeves.gt(0).sum()),
        "missing_entry_sleeves": int(weeks.missing_entry_sleeves.sum()),
        "missing_exit_sleeves": int(weeks.missing_exit_sleeves.sum()),
    }


def compound(weekly):
    return float((1 + weekly).prod() - 1) if len(weekly) and weekly.notna().mean() >= MIN_COVERAGE else np.nan


def extra_features(previous, market, origin):
    year_ago = origin - pd.DateOffset(years=1)
    half_year_ago = origin - pd.DateOffset(months=6)
    older = WeeklyEvaluation.window(previous, origin - pd.DateOffset(years=2), year_ago)
    recent = WeeklyEvaluation.window(previous, year_ago, origin)
    latest = WeeklyEvaluation.window(previous, half_year_ago, origin)
    recent_market = market.reindex(recent.index)
    return {
        "factor_previous_12m": compound(older.net), "factor_latest_12m": compound(recent.net),
        "factor_latest_6m": compound(latest.net), "index_previous_12m": compound(market.reindex(older.index)),
        "index_latest_12m": compound(recent_market), "index_latest_6m": compound(market.reindex(latest.index)),
        "index_latest_12m_week_volatility": recent_market.std(ddof=0),
        "latest_12m_mean_week_excess": (recent.net - recent_market).mean(),
    }


def collect(evaluation, candidates, market):
    window_audit, yearly_audit, rows = [], [], []
    origins = (*training_origins(), *map(pd.Timestamp, REVIEW))
    for candidate in candidates:
        weekly = evaluation.weekly(candidate, audit=True)
        details = {"candidate_id": candidate.candidate_id, "family": candidate_family(candidate)}
        for year in range(2010, 2026):
            window = evaluation.window(weekly, f"{year}-01-01", f"{year + 1}-01-01")
            yearly_audit.append({**details, "year": year, **coverage(window)})
        for origin in origins:
            end = origin + pd.DateOffset(months=WINDOW_MONTHS)
            beginning = origin - pd.DateOffset(months=WINDOW_MONTHS)
            past = evaluation.window(weekly, beginning, origin)
            future = evaluation.window(weekly, origin, end)
            key = {**details, "origin": origin.date().isoformat(), "target_end_exclusive": end.date().isoformat()}
            past_coverage, future_coverage = coverage(past), coverage(future)
            for role, counts in (("features", past_coverage), ("target", future_coverage)):
                window_audit.append({**key, "role": role, **counts})
            if not past_coverage["eligible"] or (end <= pd.Timestamp(TRAIN_END) and not future_coverage["eligible"]):
                continue
            inputs, target = features(candidate, past), performance(future)
            rows.append({**key, **inputs, **extra_features(past, market, origin),
                         "target": target["annualized_net"] if target is not None else None})
    return pd.DataFrame(rows), pd.DataFrame(window_audit), pd.DataFrame(yearly_audit)


def audit_quotes(evaluation, candidates, source, calendar):
    requests = []
    for family, date in zip(FAMILIES, AUDIT_DATES):
        candidate = next((item for item in candidates if candidate_family(item) == family), None)
        if candidate is None:
            continue
        weekly = evaluation.weekly(candidate, audit=True).loc[date:]
        ranked = evaluation.ranked_rows(candidate)
        groups = {
            "realized": weekly.net.notna(),
            "missing_exit": weekly.finite_scores.ge(TOP_K) & weekly.missing_exit_sleeves.gt(0),
        }
        for kind, eligible in groups.items():
            if not eligible.any():
                continue
            entry = eligible[eligible].index[0]
            picks = ranked.loc[ranked.entry.eq(entry)].head(TOP_K)
            requests.append((candidate, kind, entry, weekly.loc[entry, "exit"], weekly.loc[entry, "net"], picks))
    if not requests:
        raise ValueError("No measurable raw-price audit samples")
    quotes = load_quotes(source, {ticker for *_, picks in requests for ticker in picks.ticker}, calendar,
                         min(entry for _, _, entry, _, _, _ in requests),
                         max(exit_ for _, _, _, exit_, _, _ in requests))
    opening = quotes.set_index(["date", "ticker"]).open
    config = CNConfig(TOP_K)
    fees = config.commission + config.slippage
    checks = []
    for candidate, kind, entry, exit_, expected, picks in requests:
        total = 0.
        filled, missing_exits = [], []
        for ticker in picks.ticker:
            buy = opening.get((entry, ticker), np.nan)
            if pd.isna(buy) or buy <= 0:
                continue
            filled.append(ticker)
            sell = opening.get((exit_, ticker), np.nan)
            if pd.isna(sell) or sell <= 0:
                missing_exits.append(ticker)
                continue
            units = (1 / TOP_K) / (float(buy) * (1 + fees))
            total += units * float(sell) * (1 - fees - stamp_duty(exit_)) - 1 / TOP_K
        expected_fills = set(picks.loc[picks[ENTRY_OPEN].gt(0), "ticker"])
        expected_missing = set(picks.loc[picks[ENTRY_OPEN].gt(0) & ~picks[TARGET].gt(-1), "ticker"])
        error = None if missing_exits or pd.isna(expected) else float(abs(total - expected))
        passed = (set(filled) == expected_fills and set(missing_exits) == expected_missing
                  and ((pd.isna(expected) and bool(missing_exits)) or (error is not None and error <= 1e-6)))
        checks.append({"candidate_id": candidate.candidate_id, "entry": entry.date().isoformat(),
                       "exit": exit_.date().isoformat(), "kind": kind, "filled": len(filled),
                       "raw_missing_exit_tickers": missing_exits, "panel_net": None if pd.isna(expected) else float(expected),
                       "raw_net": None if missing_exits else float(total), "absolute_error": error,
                       "passed": bool(passed)})
    if not all(row["passed"] for row in checks):
        raise ValueError("Weekly labels disagree with independent raw-price calculation")
    return checks


def comparison(samples, xgb):
    train = samples.loc[samples.target_end_exclusive.le(TRAIN_END) & samples.target.notna()]
    review = samples.loc[samples.origin.isin(REVIEW)]
    selections, summaries, timings = [], [], []
    for scope in ("all_candidates", "held_out_families"):
        is_held_out = scope == "held_out_families"
        fit = train.loc[~train.family.isin(HELD_OUT)] if is_held_out else train
        pool = review.loc[review.family.isin(HELD_OUT)] if is_held_out else review
        if fit.origin.nunique() < 3 or fit.family.nunique() < 2 or set(pool.origin) != set(REVIEW):
            raise ValueError(f"Insufficient training or retrospective evaluation coverage for {scope}")
        if is_held_out and set(fit.family) & set(pool.family):
            raise ValueError("Held-out family appeared in model training")
        predicted = pool.copy()
        predicted["history"] = pool.past_annualized_net
        for method, columns in (("xgboost_v1", FEATURES), ("xgboost_extended", (*FEATURES, *ADDITIONAL_FEATURES))):
            model = xgb.XGBRegressor(**MODEL_PARAMS)
            model.fit(fit[list(columns)], fit.target)
            start = perf_counter()
            predicted[method] = model.predict(pool[list(columns)])
            timings.append({"scope": scope, "method": method, "scored": len(pool),
                            "predict_seconds": perf_counter() - start})
        for origin, group in predicted.groupby("origin", sort=True):
            for method in METHODS:
                top = group.sort_values([method, "candidate_id"], ascending=[False, True]).head(10)
                realized = top.target.dropna()
                summaries.append({"scope": scope, "origin": origin, "method": method, "train_rows": len(fit),
                                  "scored": len(group), "measured": int(group.target.notna().sum()),
                                  "top_k": len(top), "measured_top_k": len(realized),
                                  "mean_realized_annualized_net": float(realized.mean()) if len(realized) else None})
                for rank, row in enumerate(top.to_dict("records"), 1):
                    selections.append({"scope": scope, "origin": origin, "method": method, "rank": rank,
                                       "candidate_id": row["candidate_id"], "family": row["family"],
                                       "score": row[method], "realized_annualized_net": row["target"]})
    return summaries, selections, timings


def summarize_training_coverage(audit, years):
    eligible = audit.loc[audit.target_end_exclusive.le(TRAIN_END)].pivot(
        index=["origin", "candidate_id"], columns="role", values="eligible"
    )
    totals = years.loc[years.year.lt(pd.Timestamp(TRAIN_END).year)].select_dtypes("number").drop(columns="year").sum()
    return {
        "factor_windows": len(eligible), "accepted": int((eligible.features & eligible.target).sum()),
        "feature_window_only_failed": int((~eligible.features & eligible.target).sum()),
        "target_window_only_failed": int((eligible.features & ~eligible.target).sum()),
        "both_failed": int((~eligible.features & ~eligible.target).sum()),
        "candidate_weeks_before_review": {key: int(value) for key, value in totals.items() if key != "coverage"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DATA)
    args = parser.parse_args()
    import xgboost as xgb

    prices = args.source / CSI300_DATA_FILES["prices"]
    calendar_path = args.source / CSI300_DATA_FILES["calendar"]
    here = Path(__file__).resolve().parent
    pinned = [file_fingerprint(path) for path in (
        prices, calendar_path, args.panel, args.candidate_file,
        Path(__file__), here / "factor_return_model.py", here / "alphax/factor_dsl.py",
        here / "run_cn_portfolio_research.py", here / "cn_portfolio.py",
    )]
    args.output_dir.mkdir(parents=True)
    protocol = {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "inputs_and_code": pinned,
        "policy": POLICY, "minimum_week_coverage": MIN_COVERAGE, "train_end_exclusive": TRAIN_END,
        "target": "24_month_annualized_net_return", "review_origins": REVIEW, "end_exclusive": END,
        "methods": METHODS, "parameters": MODEL_PARAMS, "base_features": FEATURES,
        "additional_features": ADDITIONAL_FEATURES, "xgboost_version": xgb.__version__,
        "families": FAMILIES, "held_out_families": HELD_OUT, "quote_audit_dates": AUDIT_DATES,
        "metric": "mean realized annualized net return of top 10",
        "missing_top_results": "retain original ranks and report measured count",
        "acceptance": "outperform history on future untouched 24-month windows at comparable coverage",
        "review_scope": "retrospective comparison; fresh acceptance not evaluated",
    }
    save_json(args.output_dir / "protocol.json", protocol)

    start = perf_counter()
    calendar = read_calendar(calendar_path)
    frame = pd.read_csv(args.panel, parse_dates=["date", ENTRY, EXIT]).sort_values(["date", "ticker"]).reset_index(drop=True)
    candidates = read_candidates(args.candidate_file, frame)
    evaluation = WeeklyEvaluation(frame, calendar, END)
    quotes = load_quotes(prices, set(), calendar, START, END)
    opening = quotes.loc[quotes.ticker.eq("SH000300")].set_index("date").open
    market = evaluation.anchors.exit.map(opening) / evaluation.anchors.index.map(opening) - 1
    if market.notna().sum() == 0:
        raise ValueError("CSI300 benchmark has no measurable weekly observations")
    rows, audit, years = collect(evaluation, candidates, market)
    checks = audit_quotes(evaluation, candidates, prices, calendar)
    preprocessing = perf_counter() - start
    reports, selections, timings = comparison(rows, xgb)
    if any(file_fingerprint(Path(item["path"])) != item for item in pinned):
        raise ValueError("Review input or code changed during the run")
    audit.to_csv(args.output_dir / "coverage_windows.csv", index=False)
    years.to_csv(args.output_dir / "coverage_years.csv", index=False)
    rows.to_csv(args.output_dir / "return_samples.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output_dir / "top10.csv", index=False)
    save_json(args.output_dir / "quote_audit.json", checks)
    summary = {"candidate_count": len(candidates), "timings": timings, "preprocessing_seconds": preprocessing,
               "training_coverage": summarize_training_coverage(audit, years), "comparisons": reports,
               "quote_samples": len(checks), "latest_calendar_session": str(calendar[-1].date()),
               "fresh_acceptance": "pending; no post-freeze 24-month labels in input"}
    save_json(args.output_dir / "summary.json", summary)
    print(f"Reviewed {len(candidates)} candidates and checked {len(checks)} weeks against raw prices.")


if __name__ == "__main__":
    main()
