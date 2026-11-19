"""Compare 1/5/20-session Miner/Screener/Trader research; never reads holdout performance."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from admission import HORIZON_RESEARCH_POLICIES, build_research_pool, research_failures, strategy_plan
from cn_portfolio import simulate_cn
from medium_frequency import (TRADER_HORIZONS, anchor_mask, daily_rank_ic, factor_panel,
                              load_member_bars, screen_signals)
from miner_lite import evaluate_candidate_score, evaluate_score, load_candidates, metric_dict
from run_cn_portfolio_research import load_quotes
from run_combination_metrics import score_portfolio, top_observations
from temporal import is_feature_column
from trader_harness import (WEEKLY_COMBINATION_RISK, frequency_presets, requires_weekly_stress,
                            select_frequency, strategy_is_feasible)
from trader_lite import file_fingerprint, summarize
from weekly_xgboost import fit_weekly_scorer

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"


def miner_candidate_source(directory, cutoff):
    """Use the proposal inventory, then apply the matching-horizon research gate."""
    registry_path = directory / "registry.json"
    candidate_path = directory / "candidates.json"
    registry = json.loads(registry_path.read_text())
    generator = registry.get("generator", "")
    if not isinstance(generator, str):
        raise ValueError("Miner registry has an invalid generator")
    if (registry.get("registry_version") != "miner-lite-v3"
            or not (generator in {"external_json", "template_v1"}
                    or generator.startswith(("llm_cluster:", "llm_langgraph:", "llm_feedback_search:")))):
        raise ValueError("Expected Miner output from the JSON, template, or LLM generator")
    as_of = pd.Timestamp(registry["as_of"])
    if pd.isna(as_of) or as_of > pd.Timestamp(cutoff):
        raise ValueError("Miner candidate cutoff is after the frequency research cutoff")
    entries = json.loads(candidate_path.read_text())
    if not isinstance(entries, list) or len(entries) != registry["candidate_count"]:
        raise ValueError("Miner candidate inventory differs from its registry")
    return candidate_path, registry_path, generator, entries


CORE_PATH = HERE / "candidates_four_factor_core_v1.json"
WEEKLY_XGB_ID = "core4_xgboost"
WEEKLY_COMBINATION_WINDOWS = {"train": ("2019-01-01", "2020-12-31"),
                              "validation": ("2021-01-01", "2021-12-31")}
WEEKLY_COMBINATION_RESEARCH = replace(HORIZON_RESEARCH_POLICIES[5], min_rank_icir=.09)


def combination_research(frame, definitions, score_columns, calendar, first, windows, topk=30, *, seeded_llm=False):
    """Evaluate intact four-factor scores; constituent IC gates are independent."""
    core = [row["candidate_id"] for row in json.loads(CORE_PATH.read_text())]
    by_id = {row["candidate_id"]: row for row in definitions}
    if not set(core).issubset(by_id):
        return [], {}
    # Prevent an unrelated expression from impersonating a frozen seed ID.
    if any(by_id[name]["features"] != [name] or
           by_id[name]["expression"] != {"op": "feature", "feature": name} for name in core):
        raise ValueError("Miner core seed expressions differ from the frozen four-factor recipe")
    recipes = {"core4": core}
    for row in definitions:
        control, name = row.get("control_id"), row["candidate_id"]
        if control in core and name not in core:
            recipes[f"core4_replace_{name}"] = [name if member == control else member for member in core]
    features = {name: frame[column] for name, column in score_columns.items()}
    scores = {name: score_portfolio(frame, features, members) for name, members in recipes.items()}
    assessments = []
    for h in TRADER_HORIZONS:
        metric_windows = WEEKLY_COMBINATION_WINDOWS if h == 5 else windows
        policy = WEEKLY_COMBINATION_RESEARCH if h == 5 else HORIZON_RESEARCH_POLICIES[h]
        anchors = anchor_mask(frame.date, calendar, first, h)
        target = frame[f"target_open_open_{h}d"]
        maturity = frame[f"label_end_{h}d"]
        for name, score in scores.items():
            metrics, failures = {}, {}
            training_top = None
            for split, (start, end) in metric_windows.items():
                mask = anchors & frame.date.between(start, end) & maturity.le(pd.Timestamp(end))
                metric = metric_dict(evaluate_score(frame.date, score, target, mask, frame.ticker))
                top = top_observations(frame.loc[mask, ["date", "ticker"]], score.loc[mask], target.loc[mask], topk)
                if split == "train":
                    training_top = top
                metric["turnover"] = top.replacement.mean()
                returns = top.top_return.dropna()
                metric["top_tail_mean"] = returns.nsmallest(max(1, (len(returns) + 9) // 10)).mean()
                metrics[split] = metric
                failures[split] = research_failures(metric, 1, policy)
                if not np.isfinite(metric["top_tail_mean"]):
                    failures[split].append("missing_top_portfolio_tail")
            paired_training = None
            if seeded_llm and h == 5 and name != "core4":
                start, end = metric_windows["train"]
                mask = anchors & frame.date.between(start, end) & maturity.le(pd.Timestamp(end))
                subset = frame.loc[mask]
                common = score.notna() & scores["core4"].notna()
                core_ic = daily_rank_ic(subset, scores["core4"].where(common).loc[mask], target.loc[mask])
                candidate_ic = daily_rank_ic(subset, score.where(common).loc[mask], target.loc[mask])
                delta = (candidate_ic - core_ic).dropna().sort_index()
                blocks = np.array_split(delta.to_numpy(), 3)
                deltas = [float(block.mean()) if len(block) >= 8 else None for block in blocks]
                core_top = top_observations(subset[["date", "ticker"]], scores["core4"].loc[mask], target.loc[mask], topk)
                top_returns = pd.concat([core_top.top_return.rename("core"), training_top.top_return.rename("variant")],
                                        axis=1).dropna()
                tail_count = max(1, (len(top_returns) + 9) // 10)
                tail_delta = top_returns.variant.nsmallest(tail_count).mean() - top_returns.core.nsmallest(tail_count).mean()
                churn_delta = (training_top.replacement - core_top.replacement).dropna().mean()
                paired_training = {"fold_rank_ic_deltas": deltas, "tail_return_delta": tail_delta,
                                   "top_list_replacement_delta": churn_delta}
                if sum(value is not None and value > 0 for value in deltas) < 2:
                    failures["train"].append("no_rank_ic_improvement_in_majority_of_folds")
                if not np.isfinite(tail_delta) or tail_delta < 0:
                    failures["train"].append("worse_top_portfolio_tail")
                if not np.isfinite(churn_delta) or churn_delta > .02:
                    failures["train"].append("top_list_replacement_increase_above_two_points")
            assessments.append({"strategy_id": name, "horizon": h, "members": recipes[name],
                                "research_eligible": not any(failures.values()),
                                "metrics": metrics, "failed_checks": failures,
                                "windows": metric_windows, "policy": asdict(policy),
                                "llm_paired_training": paired_training})
    return assessments, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--miner-dir", type=Path, required=True, help="Completed Miner directory with candidates.json and registry.json")
    parser.add_argument("--feature-panel", type=Path, help="Date/ticker feature panel for Miner expressions using additional input columns")
    parser.add_argument("--evaluate-strategies", action="store_true", help="Compare daily/weekly/monthly configurations on already-examined 2022–2023; no holdout evaluation")
    parser.add_argument("--compare-weekly-xgboost", action="store_true",
                        help="Compare frozen XGBoost and equal-weight scores on the same four weekly factors")
    args = parser.parse_args()
    if args.compare_weekly_xgboost and not args.evaluate_strategies:
        parser.error("--compare-weekly-xgboost requires --evaluate-strategies")
    if args.compare_weekly_xgboost:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            parser.error(f"Install xgboost for --compare-weekly-xgboost: {exc}")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("Use a new output directory")
    source = ML / "data/raw/qlib_csi300_quantaalpha"
    # Only the research/development window is loaded unless strategy evaluation
    # is requested. 2024+ raw prices/features/labels are never loaded here.
    visible_end = "2023-12-31" if args.evaluate_strategies else "2021-12-31"
    calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
    first_execution = calendar[calendar >= "2022-01-01"][0]
    presets = frequency_presets()
    candidate_path, registry_path, generator, candidate_defs = miner_candidate_source(args.miner_dir, "2021-12-31")
    out.mkdir(parents=True)

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def write(name, data):
        (out / name).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    write("status.json", {"status": "preparing"})
    inputs = [source / "daily_pv.h5", source / "qlib_csi300.txt", source / "qlib_day.txt",
              Path(__file__), HERE / "medium_frequency.py", HERE / "cn_portfolio.py", HERE / "admission.py",
              HERE / "miner_lite.py", HERE / "trader_lite.py", HERE / "trader_harness.py",
              HERE / "run_cn_portfolio_research.py"]
    inputs.extend([candidate_path, registry_path, CORE_PATH, HERE / "run_combination_metrics.py"])
    if args.compare_weekly_xgboost:
        inputs.append(HERE / "weekly_xgboost.py")
    if args.feature_panel:
        inputs.append(args.feature_panel)
    hashes = {str(p): file_fingerprint(p) for p in inputs}
    manifest = {"scope": "retrospective_medium_frequency_research_not_live_execution",
                "created_at": datetime.now(timezone.utc).isoformat(), "sources": hashes,
                "definitions": candidate_defs, "miner_generator": generator, "horizons": list(TRADER_HORIZONS), "visible_end": visible_end,
                "label_mode": "execution_open_open", "label_rule": "T-close signal; entry T+1 open; exit T+h+1 open",
                "research_train": ["2010-01-01", "2018-12-31"], "research_development": ["2019-01-01", "2021-12-31"],
                "strategy_iteration": ["2022-01-01", "2023-12-31"], "final_reserved": ["2024-01-01", "2025-12-31"],
                "strategy_iteration_previously_examined": True,
                "research_sampling": "fixed nonoverlapping horizon anchors aligned to the frozen execution calendar phase",
                "screening": {"lookback_completed_periods": 12, "minimum_completed_periods": 6, "min_mean_rank_ic": 0.},
                "presets": {str(h): asdict(c) for h, c in presets.items()}, "holdout_evaluated": False,
                "limitations": ["5/20 sessions approximate weekly/monthly intervals, not civil-calendar weeks/months",
                                "Supplied member/calendar/adjustment provenance not independently PIT certified",
                                "Adjusted fractional units and approximate opening-gap filters; no actual lots, depth or ST limit reconstruction",
                                "Exposure is a target; holding locks/blocked exits and market drift can exceed it, while new buys have an exposure budget",
                                "Phase and terminal wind-down are frozen; factor rank turnover is not portfolio turnover",
                                "Research admission remains exploratory; 2022–2023 strategy results are iteration data, not fresh independent validation"]}
    write("manifest.json", manifest)
    write("definitions_frozen.json", candidate_defs)
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for p in inputs:
        if p.suffix == ".py":
            shutil.copyfile(p, snapshots / p.name)
    try:
        bars, intervals = load_member_bars(source / "daily_pv.h5", source / "qlib_csi300.txt", "2010-01-01", visible_end)
        parts, expected = [], 0
        groups = bars.groupby("ticker", sort=True)
        for i, (ticker, raw) in enumerate(groups, 1):
            values = factor_panel(raw, calendar, end=visible_end, horizons=TRADER_HORIZONS)
            membership = intervals[intervals.ticker == ticker]
            keep = pd.Series(False, index=values.index)
            for row in membership.itertuples():
                keep |= values.date.between(row.start, row.end)
            keep &= values.date.between("2010-01-01", visible_end)
            parts.append(values.loc[keep])
            if i % 100 == 0:
                print(f"Prepared {i}/{len(groups)} medium-frequency stock histories", flush=True)
        frame = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        del bars, parts
        sessions = calendar[(calendar >= "2010-01-01") & (calendar <= visible_end)]
        for _, group in intervals.groupby("ticker"):
            mask = np.zeros(len(sessions), dtype=bool)
            for row in group.itertuples():
                mask |= (sessions >= row.start) & (sessions <= row.end)
            expected += int(mask.sum())
        if frame.duplicated(["date", "ticker"]).any() or len(frame) != expected:
            raise ValueError("Medium-frequency panel does not match supplied historical member rows")
        required = {feature for entry in candidate_defs for feature in entry["features"]}
        if any(not is_feature_column(feature) or feature.startswith("__miner_score_") for feature in required):
            raise ValueError("Miner expressions must use feature columns, never labels or internal scores")
        missing = sorted(required - set(frame.columns))
        if missing:
            if args.feature_panel is None:
                raise ValueError("Miner expressions require --feature-panel columns: " + ", ".join(missing))
            external = pd.read_csv(args.feature_panel, usecols=["date", "ticker", *missing], parse_dates=["date"])
            external = external.loc[external.date <= pd.Timestamp(visible_end)]
            frame = frame.merge(external, on=["date", "ticker"], how="left", validate="one_to_one")
        candidates = load_candidates(candidate_path, sorted(required))
        write("feature_audit.json", {"rows": len(frame), "expected_membership_rows": expected,
                                     "max_visible_signal_date": str(frame.date.max().date()), "candidate_inputs": sorted(required)})
        write("status.json", {"status": "researching_factors"})
        results = []
        score_columns = {}
        for index, candidate in enumerate(candidates):
            name = candidate.candidate_id
            column = f"__miner_score_{index}"
            frame[column] = evaluate_candidate_score(candidate, frame)
            score_columns[name] = column
            metrics = {}
            for h in TRADER_HORIZONS:
                anchors = anchor_mask(frame.date, calendar, first_execution, h)
                metrics[str(h)] = {}
                for split, start, end in (("train", "2010-01-01", "2018-12-31"), ("validation", "2019-01-01", "2021-12-31")):
                    label_end = frame[f"label_end_{h}d"]
                    mask = anchors & frame.date.between(start, end) & label_end.notna() & label_end.le(pd.Timestamp(end))
                    m = metric_dict(evaluate_score(frame.date, frame[column], frame[f"target_open_open_{h}d"], mask, frame.ticker))
                    m["sampled_signal_days"] = int(frame.loc[mask, "date"].nunique())
                    metrics[str(h)][split] = m
            results.append({"candidate_id": name, "horizons": metrics})
            print(f"Evaluated {name}", flush=True)
        audit = {"as_of": "2021-12-31", "label_mode": "execution_open_open",
                 "windows": {"train": ["2010-01-01", "2018-12-31"], "validation": ["2019-01-01", "2021-12-31"]}}
        write("research_temporal_audit.json", audit)
        pool, assessments = build_research_pool(
            results, candidate_defs, audit, "cn", horizons=TRADER_HORIZONS,
            policies_by_horizon={h: HORIZON_RESEARCH_POLICIES[h] for h in TRADER_HORIZONS},
        )
        for entries in pool["candidates_by_horizon"].values():
            for entry in entries:
                entry["score_column"] = score_columns[entry["candidate_id"]]
        pool["miner_generator"] = generator
        combinations, combination_scores = combination_research(
            frame, candidate_defs, score_columns, calendar, first_execution, audit["windows"],
            seeded_llm=generator.endswith(":seeded"))
        pool["combinations_by_horizon"] = {
            str(h): [row["strategy_id"] for row in combinations if row["horizon"] == h and row["research_eligible"]]
            for h in TRADER_HORIZONS
        }
        write("combination_assessments.json", combinations)
        frame.to_csv(out / "features_labels.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
        write("factor_metrics.json", results)
        write("research_pool.json", pool)
        write("research_assessments.json", assessments)
        write("strategy_validation_plan.json", strategy_plan(pool, "2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"))
        write("strategy_frozen.json", {"research_pool": file_fingerprint(out / "research_pool.json"),
                                        "presets": manifest["presets"], "screening": manifest["screening"],
                                        "iteration_window": manifest["strategy_iteration"], "reserved_window": manifest["final_reserved"],
                                        "weekly_combination_calibration": {"windows": WEEKLY_COMBINATION_WINDOWS,
                                            "research_policy": asdict(WEEKLY_COMBINATION_RESEARCH),
                                            "risk_thresholds_both_costs": WEEKLY_COMBINATION_RISK,
                                            "calibrated_on_observed_data": True}})
        trials = []
        weekly_comparison = None
        if args.evaluate_strategies:
            write("status.json", {"status": "evaluating_frozen_strategy_iterations"})
            start, end = manifest["strategy_iteration"]
            previous = calendar[calendar.get_loc(first_execution) - 1]
            if args.compare_weekly_xgboost:
                core = next((item for item in combinations
                             if item["horizon"] == 5 and item["strategy_id"] == "core4"), None)
                if core is None:
                    raise ValueError("The weekly XGBoost comparison requires all four frozen core factors in the Miner inventory")
                columns = [score_columns[name] for name in core["members"]]
                scores, model, details = fit_weekly_scorer(frame, columns, previous, end)
                model.save_model(out / "weekly_xgboost.ubj")
                write("weekly_xgboost_model.json", {**details, "members": dict(zip(columns, core["members"])),
                                                   "model": file_fingerprint(out / "weekly_xgboost.ubj")})
                weekly_comparison = {"window": [start, end], "models": {}}
            for h, config in presets.items():
                strategies = []
                diagnostics = set()
                if pool["candidates_by_horizon"][str(h)]:
                    signals, selections = screen_signals(frame, pool, h, calendar, first_execution, start, end)
                    if not signals.empty and np.isfinite(signals.prediction).any():
                        strategies.append(("screened", None, signals))
                        selections.to_csv(out / f"screening_{h}d.csv", index=False)
                for item in combinations:
                    if item["horizon"] != h or not item["research_eligible"]:
                        continue
                    name = item["strategy_id"]
                    mask = anchor_mask(frame.date, calendar, first_execution, h) & frame.date.between(previous, end)
                    signals = frame.loc[mask, ["date", "ticker"]].assign(prediction=combination_scores[name].loc[mask])
                    if np.isfinite(signals.prediction).any():
                        strategies.append((name, item["members"], signals))
                if h == 5 and weekly_comparison is not None:
                    mask = anchor_mask(frame.date, calendar, first_execution, h) & frame.date.between(previous, end)
                    weekly = frame.loc[mask, ["date", "ticker"]]
                    if not any(name == "core4" for name, _, _ in strategies):
                        diagnostics.add("core4")
                        strategies.append(("core4", core["members"],
                                           weekly.assign(prediction=combination_scores["core4"].loc[mask])))
                    diagnostics.add(WEEKLY_XGB_ID)
                    strategies.append((WEEKLY_XGB_ID, core["members"], weekly.assign(prediction=scores.reindex(weekly.index))))
                if not strategies:
                    trials.append({"horizon": h, "status": "not_run_no_research_signals"})
                    continue
                tickers = set().union(*(set(signals.ticker) for _, _, signals in strategies))
                quotes = load_quotes(source / "daily_pv.h5", tickers, calendar, start, end)
                for name, members, signals in strategies:
                    tag = f"{h}d" if name == "screened" else f"{h}d_{name}"
                    signals.to_csv(out / f"signals_{tag}.csv.gz", index=False)
                    report, trades, positions, execution_audit = simulate_cn(signals, quotes, calendar, config, start, end)
                    metrics = summarize(report, 1_000_000., annual_risk_free_rate=0.)
                    trial = {"horizon": h, "strategy_id": name, "members": members,
                             "config": asdict(config), **metrics, "execution_audit": execution_audit,
                             "paid_cost_fraction_of_initial_capital": float(trades.total_cost.sum())}
                    if requires_weekly_stress(h, members):
                        stress_config = replace(config, commission=2 * config.commission, slippage=2 * config.slippage)
                        stressed, _, _, stress_audit = simulate_cn(signals, quotes, calendar, stress_config, start, end)
                        stressed.to_csv(out / f"returns_{tag}_double_cost.csv")
                        trial["double_cost"] = {**summarize(stressed, 1_000_000., annual_risk_free_rate=0.),
                                                "execution_audit": stress_audit}
                    if name in diagnostics:
                        trial["status"] = "diagnostic_only"
                    else:
                        trial["status"] = "feasible_in_proxy" if strategy_is_feasible(trial) else "not_feasible"
                    if h == 5 and weekly_comparison is not None and name in {"core4", WEEKLY_XGB_ID}:
                        weekly_comparison["models"][name] = {
                            **{key: trial[key] for key in ("cumulative_return", "max_drawdown", "sharpe", "mean_daily_turnover")},
                            "double_cost_cumulative_return": trial["double_cost"]["cumulative_return"],
                        }
                        weekly_comparison["csi300_price_index_return"] = float((1 + report.benchmark_return).prod() - 1)
                    trials.append(trial)
                    report.to_csv(out / f"returns_{tag}.csv")
                    trades.to_csv(out / f"trades_{tag}.csv", index=False)
                    positions.to_csv(out / f"positions_{tag}.csv.gz", index=False)
        for p, expected_hash in hashes.items():
            if file_fingerprint(Path(p)) != expected_hash:
                raise ValueError("Input/code changed: " + p)
        summary = {"research_eligible": pool["eligible_counts"], "eligible_combinations": pool["combinations_by_horizon"], "strategies": trials, "holdout_evaluated": False,
                   "strategy_window_is_previously_examined": True, "scope": manifest["scope"]}
        if weekly_comparison is not None:
            summary["weekly_xgboost_comparison"] = weekly_comparison
        selection = select_frequency(trials, manifest["strategy_iteration"], evaluated=args.evaluate_strategies)
        write("strategy_selection.json", {**selection, "market": "cn", "label_mode": pool["label_mode"],
                                           "frozen_as_of": manifest["strategy_iteration"][1],
                                           "screening": manifest["screening"],
                                           "research_pool_sha256": file_fingerprint(out / "research_pool.json")["sha256"]})
        write("summary.json", summary)
        write("status.json", {"status": "completed_research"})
        lines = ["# Medium-frequency factor and strategy research", "",
                 f"Research eligibility: {pool['eligible_counts']}. Reserved 2024–2025 not evaluated.", "",
                 "Labels: next-open entry to open after 1/5/20 holding sessions; research uses fixed nonoverlapping anchors.",
                 "Research 2010–2018 / development 2019–2021; strategy iterations 2022–2023 were previously examined.", "",
                 "| Holding horizon | Status | Net cumulative | Annualized | Max drawdown | Mean daily turnover |", "|---|---|---:|---:|---:|---:|"]
        for trial in trials:
            if "cumulative_return" in trial:
                lines.append(f"| {trial['horizon']}d | {trial['status']} | {trial['cumulative_return']:.2%} | {trial['annualized_return']:.2%} | {trial['max_drawdown']:.2%} | {trial['mean_daily_turnover']:.2%} |")
            else:
                lines.append(f"| {trial['horizon']}d | {trial['status']} | — | — | — | — |")
        lines += ["", "Budget limits include initial formation and terminal wind-down; locked or untradable inventory is never erased.",
                  "Costs, signal definitions, screening rules and source hashes were frozen before strategy iteration.", "", *["- " + line for line in manifest["limitations"]]]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
