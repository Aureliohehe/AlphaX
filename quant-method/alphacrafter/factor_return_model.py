"""Train and use a shared XGBoost to rank Miner factors by two-year net return."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphax.factor_dsl import (
    candidate_expression_spec, candidate_from_dict, evaluate_candidate_score,
)
from alphax.experiment import CSI300_DATA_FILES
from cn_portfolio import CNConfig, stamp_duty
from csi300_data import member_mask
from medium_frequency import FACTOR_HYPOTHESES, factor_panel, load_member_bars
from miner_lite import BASELINE_FEATURES, load_candidates
from run_cn_portfolio_research import load_quotes
from temporal import is_feature_column
from trader_lite import file_fingerprint

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "machine-learn" / "data/raw/qlib_csi300_quantaalpha"
START, TRAIN_END, END = "2010-01-01", "2022-01-01", "2026-01-01"
REVIEW = ("2022-01-01", "2024-01-01")
HOLD_SESSIONS = 5
WINDOW_MONTHS = 24
TOP_K = 30
MIN_COVERAGE = .9
ENTRY = "label_start_5d"
EXIT = "label_end_5d"
TARGET = "target_open_open_5d"
ENTRY_OPEN = "target_entry_open_5d"
FEATURES = (
    "past_annualized_net", "past_mean_week", "past_week_volatility", "past_worst_week",
    "past_drawdown", "past_positive_share", "past_observed_share", "past_filled_share",
    "ast_nodes", "ast_max_window", "input_count",
)
POLICY = "csi300_five_session_top30_equal_weight_roundtrip_v1"
MODEL_PARAMS = {
    "objective": "reg:squarederror", "n_estimators": 120, "max_depth": 2,
    "learning_rate": .04, "min_child_weight": 12, "reg_lambda": 5.,
    "subsample": .85, "colsample_bytree": .9, "random_state": 42, "n_jobs": 4,
}


def read_calendar(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    dates = pd.DatetimeIndex(pd.to_datetime([line for line in lines if line != "date"]))
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Calendar must contain unique increasing session dates")
    return dates


def build_csi300_panel(source, calendar, end):
    bars, intervals = load_member_bars(source / CSI300_DATA_FILES["prices"],
                                       source / CSI300_DATA_FILES["membership"], START, end)
    memberships = {ticker: rows for ticker, rows in intervals.groupby("ticker")}
    pieces = []
    for ticker, raw in bars.groupby("ticker", sort=True):
        panel = factor_panel(raw, calendar, end=end, horizons=(HOLD_SESSIONS,))
        openings = raw.set_index("date").open
        panel[ENTRY_OPEN] = panel[ENTRY].map(openings).where(lambda values: np.isfinite(values) & values.gt(0))
        keep = member_mask(panel.date, memberships[ticker]) & panel.date.ge(START)
        pieces.append(panel.loc[keep])
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def default_candidates(available):
    """A reproducible pilot inventory covering native factors and trailing-window variants."""
    candidates = []
    for feature, hypothesis in FACTOR_HYPOTHESES.items():
        if feature not in available:
            continue
        for op, horizons in (("feature", (None,)), ("ts_mean", (5, 10, 20)),
                             ("ts_delta", (5, 10, 20)), ("ts_zscore", (5, 10, 20))):
            for window in horizons:
                ast = {"op": op, "feature": feature}
                if window is not None:
                    ast["window"] = window
                candidates.append(candidate_from_dict({
                    "candidate_id": feature if window is None else f"{feature}__{op}{window}",
                    "expression": ast, "hypothesis": hypothesis,
                }))
    return candidates


class WeeklyEvaluation:
    """Batch the price-based labels while keeping stock selection signal-time only."""

    def __init__(self, frame, calendar, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        self.frame = frame
        days = pd.DatetimeIndex(frame.date)
        positions = calendar.get_indexer(days)
        if (positions < 0).any() or frame.duplicated(["date", "ticker"]).any():
            raise ValueError("Panel keys must be unique and on the supplied calendar")
        # Anchor to the first 2010 market session. Keep this phase at every prediction date.
        origin = calendar.searchsorted(pd.Timestamp(START))
        if origin == len(calendar) or calendar[origin].year != 2010:
            raise ValueError("Calendar must include the fixed 2010 rebalance origin")
        anchors = (positions + 1 - origin) % HOLD_SESSIONS == 0
        in_range = (positions + 1 >= origin) & (positions + HOLD_SESSIONS + 1 < len(calendar))
        mask = anchors & in_range
        indices = positions[mask]
        expected_entry = calendar[indices + 1]
        expected_exit = calendar[indices + HOLD_SESSIONS + 1]
        settled = expected_exit < self.cutoff
        entries = pd.to_datetime(frame.loc[mask, ENTRY]).to_numpy()
        exits = pd.to_datetime(frame.loc[mask, EXIT]).to_numpy()
        if (entries[settled] != expected_entry[settled]).any() or (exits[settled] != expected_exit[settled]).any():
            raise ValueError("Weekly labels do not match exact entry/exit calendar sessions")
        self.rows = frame.loc[mask, ["date", "ticker", ENTRY_OPEN, TARGET]].loc[settled].copy()
        self.rows["entry"] = expected_entry[settled]
        self.rows["exit"] = expected_exit[settled]
        session_positions = np.arange(origin, len(calendar) - HOLD_SESSIONS, HOLD_SESSIONS)
        entry = calendar[session_positions]
        exit_ = calendar[session_positions + HOLD_SESSIONS]
        self.anchors = pd.DataFrame({"exit": exit_[exit_ < self.cutoff]}, index=entry[exit_ < self.cutoff])
        terms = CNConfig(TOP_K)
        self.side_cost = terms.commission + terms.slippage
        self.sell_tax = pd.Series([stamp_duty(day) for day in self.anchors.exit], index=self.anchors.index)

    def predict_features(self, candidate, cutoff):
        weekly = self.weekly(candidate)
        past = self.window(weekly, pd.Timestamp(cutoff) - pd.DateOffset(months=WINDOW_MONTHS), cutoff)
        return features(candidate, past)

    def ranked_rows(self, candidate):
        scores = evaluate_candidate_score(candidate, self.frame)
        rows = self.rows.assign(score=scores.reindex(self.rows.index))
        return rows.loc[np.isfinite(rows.score)].sort_values(["entry", "score", "ticker"],
                                                            ascending=[True, False, True])

    def weekly(self, candidate, *, audit=False):
        rows = self.ranked_rows(candidate)
        selected = rows.groupby("entry", sort=False).head(TOP_K)
        # Each sleeve buys with fees included and liquidates at the scheduled open.
        # Missing entry leaves that sleeve in cash. A held stock without an exit quote
        # makes the entire period unknown, so no replacement or stale price is used.
        filled = np.isfinite(selected[ENTRY_OPEN]) & selected[ENTRY_OPEN].gt(0)
        exits = np.isfinite(selected[TARGET]) & selected[TARGET].gt(-1)
        proceeds = (1 + selected[TARGET]) * (
            1 - self.side_cost - selected.entry.map(self.sell_tax)
        ) / (1 + self.side_cost) - 1
        net = proceeds.where(filled, 0.)
        groups = selected.entry
        counts = selected.groupby("entry").size()
        missing_exit = (filled & ~exits).groupby(groups).any()
        weekly = self.anchors.copy()
        weekly["net"] = (net.groupby(groups).sum() / TOP_K).where(counts.eq(TOP_K) & ~missing_exit)
        weekly["filled_share"] = filled.groupby(groups).sum() / TOP_K
        if audit:
            valid_inputs = np.isfinite(self.frame.loc[self.rows.index, list(candidate.features)]).all(axis=1)
            counts = {
                "member_rows": self.rows.groupby("entry").size(),
                "complete_inputs": valid_inputs.groupby(self.rows.entry).sum(),
                "finite_scores": rows.groupby("entry").size(),
                "missing_entry_sleeves": (~filled).groupby(groups).sum(),
                "missing_exit_sleeves": (filled & ~exits).groupby(groups).sum(),
            }
            for name, series in counts.items():
                weekly[name] = series.reindex(weekly.index, fill_value=0).astype(int)
        return weekly

    @staticmethod
    def window(weekly, start, end):
        return weekly.loc[(weekly.index >= pd.Timestamp(start)) & (weekly.exit < pd.Timestamp(end))]


def performance(window):
    if window.empty:
        return None
    values = window.net.dropna()
    observed = len(values) / len(window)
    if observed < MIN_COVERAGE:
        return None
    log_growth = np.log1p(values).sum()
    wealth = np.r_[1., (1 + values).cumprod().to_numpy()]
    return {
        "annualized_net": float(np.expm1(log_growth * (252 / HOLD_SESSIONS) / len(values))),
        "cumulative_net": float(np.expm1(log_growth)),
        "mean_week": float(values.mean()), "week_volatility": float(values.std(ddof=0)),
        "worst_week": float(values.min()), "drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "positive_share": float(values.gt(0).mean()), "observed_share": observed,
        "filled_share": float(window.loc[values.index, "filled_share"].mean()),
        "weeks": len(window), "observed_weeks": len(values),
    }


def ast_shape(ast):
    children = ([ast["input"]] if "input" in ast else ast.get("inputs", []))
    shapes = [ast_shape(child) for child in children]
    return 1 + sum(count for count, _ in shapes), max([ast.get("window", 0), *(window for _, window in shapes)])


def features(candidate, window):
    stats = performance(window)
    if stats is None:
        return None
    values = {f"past_{key}": stats[key] for key in (
        "annualized_net", "mean_week", "week_volatility", "worst_week", "drawdown",
        "positive_share", "observed_share", "filled_share",
    )}
    nodes, max_window = ast_shape(candidate_expression_spec(candidate))
    return {**values, "ast_nodes": nodes, "ast_max_window": max_window, "input_count": len(set(candidate.features))}


def read_candidates(path, frame):
    usable = [column for column in frame if is_feature_column(column) and column not in BASELINE_FEATURES]
    candidates = load_candidates(path, usable) if path else default_candidates(usable)
    if not candidates or len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Expected at least one candidate with unique IDs")
    return candidates


def training_origins():
    return [day for day in pd.date_range("2012-01-01", TRAIN_END, freq="6MS")
            if day + pd.DateOffset(months=WINDOW_MONTHS) <= pd.Timestamp(TRAIN_END)]


def samples(evaluation, candidates):
    cohorts = training_origins()
    training, review, full = [], [], []
    for candidate in candidates:
        weekly = evaluation.weekly(candidate)
        four_year = performance(evaluation.window(weekly, REVIEW[0], END))
        if four_year is not None:
            full.append({"candidate_id": candidate.candidate_id, **four_year})
        for origin in (*cohorts, *map(pd.Timestamp, REVIEW)):
            end = origin + pd.DateOffset(months=WINDOW_MONTHS)
            previous = evaluation.window(weekly, origin - pd.DateOffset(months=WINDOW_MONTHS), origin)
            following = evaluation.window(weekly, origin, end)
            inputs, observed = features(candidate, previous), performance(following)
            if inputs is None:
                continue
            row = {"candidate_id": candidate.candidate_id, "origin": origin.date().isoformat(),
                   "target_end_exclusive": end.date().isoformat(),
                   "target": observed["annualized_net"] if observed else None, **inputs}
            if end <= pd.Timestamp(TRAIN_END):
                if observed is not None:
                    training.append(row)
            else:
                row.update(realized_cumulative_net=observed["cumulative_net"] if observed else None,
                           realized_drawdown=observed["drawdown"] if observed else None,
                           realized_week_coverage=float(following.net.notna().mean()))
                for year in range(origin.year, end.year):
                    part = performance(evaluation.window(weekly, f"{year}-01-01", f"{year + 1}-01-01"))
                    row[f"net_{year}"] = part["cumulative_net"] if part else None
                review.append(row)
    return pd.DataFrame(training), pd.DataFrame(review), full


def review_scores(review, candidate_count):
    reports = {}
    for origin in REVIEW:
        group = review.loc[review.origin.eq(origin)]
        if group.empty:
            reports[origin] = {"evaluated": 0, "candidates": candidate_count}
            continue
        k = min(10, len(group))
        rankings = {}
        for method in ("prediction", "past_annualized_net"):
            selected = group.nlargest(k, method).target.dropna()
            measured = group.dropna(subset=["target"])
            correlation = measured[method].rank().corr(measured.target.rank())
            rankings[method] = {
                "spearman": float(correlation) if np.isfinite(correlation) else None,
                "mean_realized_annualized_net_top_k": float(selected.mean()) if len(selected) else None,
                "best_realized_annualized_net_top_k": float(selected.max()) if len(selected) else None,
                "top_k": k, "measured_top_k": len(selected),
            }
        reports[origin] = {"scored": len(group), "evaluated": int(group.target.notna().sum()),
                           "candidates": candidate_count, **rankings}
    return reports


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def index_benchmark(source, calendar, evaluation):
    quotes = load_quotes(source / CSI300_DATA_FILES["prices"], set(), calendar, REVIEW[0], END)
    opens = quotes.loc[quotes.ticker.eq("SH000300")].set_index("date").open
    result = {}
    for start, end in ((REVIEW[0], REVIEW[1]), (REVIEW[1], END), (REVIEW[0], END)):
        dates = evaluation.window(evaluation.anchors, start, end)
        returns = dates.exit.map(opens) / dates.index.map(opens) - 1
        result[f"{start}/{end}"] = float((1 + returns).prod() - 1) if len(dates) and returns.notna().all() else None
    return result


def train(args, calendar):
    import xgboost as xgb

    args.output_dir.mkdir(parents=True)
    if calendar[-1] < pd.Timestamp(END):
        raise ValueError("Calendar does not cover the 2022–2025 review")
    paths = [args.calendar, args.panel] if args.panel else [
        args.calendar, args.source / CSI300_DATA_FILES["prices"], args.source / CSI300_DATA_FILES["membership"],
    ]
    if args.candidate_file:
        paths.append(args.candidate_file)
    inputs = [file_fingerprint(path) for path in paths]
    code = [file_fingerprint(HERE / name) for name in (
        "factor_return_model.py", "medium_frequency.py", "cn_portfolio.py", "csi300_data.py",
        "miner_lite.py", "alphax/factor_dsl.py", "run_cn_portfolio_research.py",
    )]
    frame = (pd.read_csv(args.panel, parse_dates=["date", ENTRY, EXIT]) if args.panel
             else build_csi300_panel(args.source, calendar, END))
    frame = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    if frame.date.max() < calendar[calendar < END][-1]:
        raise ValueError("Feature panel does not cover the 2022–2025 review")
    candidates = read_candidates(args.candidate_file, frame)
    print(f"Evaluating {len(candidates)} candidate definitions on {len(frame):,} member rows", flush=True)
    evaluation = WeeklyEvaluation(frame, calendar, END)
    training, review, full = samples(evaluation, candidates)
    if (training.empty or training.origin.nunique() < 3 or training.candidate_id.nunique() < 2
            or review.empty or set(review.dropna(subset=["target"]).origin) != set(REVIEW)):
        raise ValueError("Insufficient price coverage for model training or one of the two review windows")
    benchmark = index_benchmark(args.source, calendar, evaluation) if not args.panel else {}
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(training.loc[:, list(FEATURES)], training.target)
    review["prediction"] = model.predict(review.loc[:, list(FEATURES)])
    if any(file_fingerprint(Path(item["path"])) != item for item in [*inputs, *code]):
        raise ValueError("Input or code changed during return-model training")
    training.to_csv(args.output_dir / "training_returns.csv", index=False)
    review.sort_values(["origin", "prediction"], ascending=[True, False]).to_csv(
        args.output_dir / "review_returns.csv", index=False
    )
    four_year = pd.DataFrame(full)
    four_year.to_csv(args.output_dir / "review_2022_2025.csv", index=False)
    model_path = args.output_dir / "return_model.ubj"
    model.save_model(model_path)
    if args.export_panel:
        frame.to_csv(args.output_dir / "weekly_labels.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
    save_json(args.output_dir / "candidates.json", [
        {**asdict(candidate), "expression": candidate_expression_spec(candidate)} for candidate in candidates
    ])
    metadata = {
        "policy": POLICY, "target": "24_month_annualized_net_return", "features": FEATURES,
        "train_end_exclusive": TRAIN_END, "training_rows": len(training),
        "training_cohorts": int(training.origin.nunique()), "training_factors": int(training.candidate_id.nunique()),
        "review": review_scores(review, len(candidates)), "review_status": "retrospective_already_observed",
        "four_year": {"start": REVIEW[0], "end_exclusive": END, "evaluated": len(four_year)},
        "csi300_gross_benchmark": benchmark,
        "top_k": TOP_K, "hold_sessions": HOLD_SESSIONS, "minimum_week_coverage": MIN_COVERAGE,
        "model": file_fingerprint(model_path), "inputs": inputs, "code": code,
        "calendar_phase": calendar[calendar.searchsorted(START)].date().isoformat(),
        "model_parameters": MODEL_PARAMS, "xgboost_version": xgb.__version__,
    }
    save_json(args.output_dir / "return_model.json", metadata)
    return metadata["review"]


def score(args, calendar):
    import xgboost as xgb

    meta = json.loads((args.model_dir / "return_model.json").read_text())
    cutoff = pd.Timestamp(args.as_of)
    if meta["policy"] != POLICY or tuple(meta["features"]) != FEATURES or cutoff < pd.Timestamp(meta["train_end_exclusive"]):
        raise ValueError("Return model is incompatible with the scoring policy or cutoff")
    historical_calendar = calendar[calendar >= START]
    if historical_calendar.empty or historical_calendar[0].date().isoformat() != meta["calendar_phase"]:
        raise ValueError("Scoring calendar does not use the trained rebalance phase")
    path = args.model_dir / "return_model.ubj"
    if file_fingerprint(path)["sha256"] != meta["model"]["sha256"]:
        raise ValueError("Return model fingerprint mismatch")
    frame = pd.read_csv(args.panel, parse_dates=["date", ENTRY, EXIT])
    frame = frame.loc[frame.date.lt(cutoff)].sort_values(["date", "ticker"]).reset_index(drop=True)
    if frame.empty or cutoff > calendar[-1] or frame.date.max() < calendar[calendar < cutoff][-1]:
        raise ValueError("Feature panel and calendar must cover the scoring cutoff")
    candidates = read_candidates(args.candidate_file, frame)
    evaluation = WeeklyEvaluation(frame, calendar, cutoff)
    scored, skipped = [], []
    for candidate in candidates:
        row = evaluation.predict_features(candidate, cutoff)
        if row is None:
            skipped.append(candidate.candidate_id)
        else:
            scored.append({"candidate_id": candidate.candidate_id, **row})
    if not scored:
        raise ValueError("No candidate has enough mature weekly returns at the scoring cutoff")
    model = xgb.XGBRegressor()
    model.load_model(path)
    table = pd.DataFrame(scored)
    table["predicted_annualized_net_24m"] = model.predict(table.loc[:, list(FEATURES)])
    table = table.sort_values(["predicted_annualized_net_24m", "candidate_id"], ascending=[False, True])
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "ranked_candidates.csv", index=False)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    save_json(args.output_dir / "ranked_candidates.json", [
        {**asdict(by_id[row.candidate_id]), "expression": candidate_expression_spec(by_id[row.candidate_id]),
         "predicted_annualized_net_24m": row.predicted_annualized_net_24m}
        for row in table.itertuples()
    ])
    summary = {"as_of_exclusive": args.as_of, "scored": len(table), "skipped_for_coverage": skipped,
               "model_sha256": meta["model"]["sha256"]}
    save_json(args.output_dir / "ranking.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, title="Commands")
    fit = commands.add_parser("train", help="Fit a shared model on historical factor returns")
    rank = commands.add_parser("score", help="Rank frozen candidates by predicted 24-month annualized net return")
    for command in (fit, rank):
        command.add_argument("--output-dir", required=True, type=Path, help="Directory for model or ranking artifacts")
        command.add_argument("--candidate-file", type=Path, required=command is rank, help="Frozen candidate JSON")
        command.add_argument("--panel", type=Path, required=command is rank, help="Feature panel with weekly return labels")
        command.add_argument("--calendar", type=Path, default=DATA / CSI300_DATA_FILES["calendar"],
                             help="CSI 300 trading-session calendar")
    fit.add_argument("--source", type=Path, default=DATA, help="Raw CSI 300 input directory")
    fit.add_argument("--export-panel", action="store_true", help="Export the labelled panel with the model artifacts")
    rank.add_argument("--model-dir", type=Path, required=True, help="Directory produced by the train command")
    rank.add_argument("--as-of", required=True, help="Exclusive scoring cutoff in YYYY-MM-DD format")
    args = parser.parse_args()
    try:
        if args.command == "train" and args.source != DATA and args.calendar == DATA / CSI300_DATA_FILES["calendar"]:
            args.calendar = args.source / CSI300_DATA_FILES["calendar"]
        calendar = read_calendar(args.calendar)
        result = train(args, calendar) if args.command == "train" else score(args, calendar)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
