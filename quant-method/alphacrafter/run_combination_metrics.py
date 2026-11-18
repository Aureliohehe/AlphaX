"""Paired signal and top-k metrics for Miner-sourced single-factor substitutions."""
import argparse
import json
from pathlib import Path

import pandas as pd

from medium_frequency import TRADER_HORIZONS, anchor_mask, daily_rank_ic
from miner_lite import evaluate_candidate_score, load_candidates
from temporal import is_feature_column
from trader_lite import file_fingerprint

HERE = Path(__file__).resolve().parent


def score_portfolio(frame, feature_scores, ids):
    ranks = [feature_scores[name].groupby(frame.date).rank(pct=True) for name in ids]
    return pd.concat(ranks, axis=1).sum(axis=1, min_count=len(ids)) / len(ids)


def top_observations(frame, score, target, topk):
    selected = frame.assign(score=score, target=target).dropna(subset=["score"])
    selected = selected.sort_values(["date", "score", "ticker"], ascending=[True, False, True])
    rows, previous = [], None
    for date, group in selected.groupby("date", sort=True):
        top = group.head(topk)
        if len(top) < topk:
            previous = None
            continue
        names = set(top.ticker)
        rows.append({
            "date": date,
            "top_return": float(top.target.mean()) if top.target.notna().all() else None,
            "replacement": len(names - previous) / topk if previous is not None else None,
        })
        previous = names
    return pd.DataFrame(rows, columns=["date", "top_return", "replacement"]).set_index("date")


def rank_metrics(ic):
    mean = ic.mean()
    std = ic.std()
    return {"rank_ic": clean(mean), "rank_icir": clean(mean / std if pd.notna(std) and std > 0 else None)}


def clean(value):
    return float(value) if value is not None and pd.notna(value) else None


def paired_metrics(frame, core, variant, target_column, topk, start, end):
    target = frame[target_column]
    common = core.notna() & variant.notna()
    core_ic = daily_rank_ic(frame, core.where(common), target)
    variant_ic = daily_rank_ic(frame, variant.where(common), target)
    ic = pd.concat({"core": core_ic, "variant": variant_ic}, axis=1).dropna()
    top = {name: top_observations(frame, score, target, topk)
           for name, score in (("core", core), ("variant", variant))}
    returns = pd.concat({name: table.top_return for name, table in top.items()}, axis=1).dropna()
    turnover = pd.concat({name: table.replacement for name, table in top.items()}, axis=1).dropna()
    results = {}
    for name, score in (("core", core), ("variant", variant)):
        returns_count = len(returns)
        tail_count = max(1, (returns_count + 9) // 10)
        results[name] = {
            **rank_metrics(ic[name]),
            "coverage": clean(score.notna().groupby(frame.date).mean().mean()),
            "top_tail_mean": clean(returns[name].nsmallest(tail_count).mean()) if returns_count else None,
            "top_list_replacement": clean(turnover[name].mean()),
        }
    # One-year rolling comparisons sampled every six months. Assign IC by signal date.
    windows = []
    if not frame.empty:
        for window_start in pd.date_range(start, end, freq="6MS"):
            window_end = window_start + pd.DateOffset(years=1)
            if window_end > end + pd.Timedelta(days=1):
                break
            entry_dates = frame.groupby("date").entry_date.first().reindex(ic.index)
            block = ic.loc[(entry_dates >= window_start) & (entry_dates < window_end)]
            if len(block) >= 8:
                windows.append({"start": str(window_start.date()), "end_exclusive": str(window_end.date()),
                                "observations": len(block),
                                "rank_ic_delta": float((block.variant - block.core).mean())})
    return {"metrics": results, "paired_ic_dates": len(ic), "paired_top_return_dates": len(returns),
            "paired_replacement_dates": len(turnover), "rolling_windows": windows,
            "rolling_windows_improved": sum(window["rank_ic_delta"] > 0 for window in windows)}


def main():
    from run_medium_frequency import miner_candidate_source

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miner-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True, help="One market session per line")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if args.topk < 1 or start >= end:
        parser.error("Require a positive top-k and ordered evaluation dates")
    candidate_path, registry, generator, entries = miner_candidate_source(args.miner_dir, start - pd.Timedelta(days=1))
    core_path = HERE / "candidates_four_factor_core_v1.json"
    core_ids = [row["candidate_id"] for row in json.loads(core_path.read_text())]
    variants = [row for row in entries if row.get("control_id") in core_ids]
    if not set(core_ids).issubset({row["candidate_id"] for row in entries}) or not variants:
        raise ValueError("Miner output must contain all four core candidates and at least one single-factor substitution")
    required = {feature for row in entries for feature in row["features"]}
    if any(not is_feature_column(feature) for feature in required):
        raise ValueError("Miner expressions must not reference outcome columns")
    labels = [f"{prefix}_{h}d" for h in TRADER_HORIZONS for prefix in ("target_open_open", "label_end")]
    frame = pd.read_csv(args.panel, usecols=["date", "ticker", *required, *labels], parse_dates=["date", *labels[1::2]])
    frame = frame.loc[frame.date <= end].sort_values(["date", "ticker"]).reset_index(drop=True)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate panel keys")
    calendar = pd.DatetimeIndex(pd.to_datetime(args.calendar.read_text().splitlines()))
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or not calendar[0] < start <= calendar[-1]:
        raise ValueError("Require a unique ordered calendar with a session before evaluation")
    first = calendar[calendar >= start][0]
    previous = calendar[calendar.get_loc(first) - 1]
    candidates = load_candidates(candidate_path, sorted(required))
    feature_scores = {candidate.candidate_id: evaluate_candidate_score(candidate, frame) for candidate in candidates}
    core = score_portfolio(frame, feature_scores, core_ids)
    results = {}
    for variant in variants:
        name, control = variant["candidate_id"], variant["control_id"]
        score = score_portfolio(frame, feature_scores, [name if cid == control else cid for cid in core_ids])
        comparisons = {}
        for h in TRADER_HORIZONS:
            label_end = frame[f"label_end_{h}d"]
            expected_end = frame.date.map(pd.Series(calendar, index=calendar).shift(-(h + 1)))
            if (label_end.notna() & label_end.ne(expected_end)).any():
                raise ValueError(f"{h}-session labels do not match the market calendar")
            mask = (anchor_mask(frame.date, calendar, first, h) & frame.date.between(previous, end)
                    & expected_end.le(end))
            target_column = f"target_open_open_{h}d"
            sampled = frame.loc[mask, ["date", "ticker", target_column]].copy()
            sampled[target_column] = sampled[target_column].where(label_end.loc[mask].notna())
            entries = frame.date.map(pd.Series(calendar, index=calendar).shift(-1))
            sampled["entry_date"] = entries.loc[mask]
            comparisons[str(h)] = paired_metrics(sampled, core.loc[mask], score.loc[mask],
                                                 target_column, args.topk, start, end)
        results[name] = {"control_id": control, "comparisons": comparisons}
    summary = {"scope": "paired_signal_metrics", "window": [args.start, args.end], "topk": args.topk,
               "miner_generator": generator, "core_ids": core_ids, "substitutions": results,
               "inputs": [file_fingerprint(path) for path in (candidate_path, registry, core_path, args.panel, args.calendar)]}
    with args.output.open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
