#!/usr/bin/env python3
"""Exploratory agent factor -> retrospective library -> Screener -> local intraday fills.

NOT AlphaCrafter's paper gates or a broker execution. The admission rule is a
separate, explicitly weak 1-day demo rule. Uses historical data already viewed
in other experiments, so returns cannot be presented as new out-of-sample alpha.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alphax.factor_dsl import candidate_from_dict
from trader_lite import TraderConfig, file_fingerprint, load_inputs, simulate, summarize

HERE = Path(__file__).resolve().parent
ADMISSION_RULE = "1d_train_and_validation_positive_ic_rankic_icir_rankicir_hit_above_half_coverage_above_90_turnover_below_40"
COST_RATE = 0.0001
CONFIG = TraderConfig(beta=0.1, gamma=1.0, nlong=10, nshort=0)
INITIAL_CASH = 1_000_000.0


def simulated_orders(frame: pd.DataFrame, report: pd.DataFrame, positions: pd.DataFrame,
                     cost_rate: float, initial_cash: float = INITIAL_CASH) -> pd.DataFrame:
    """Produce auditable notional-only open/close fills, not broker orders or shares.

    Features contain open-to-close returns but no executable open/close prices, so
    price and share quantity must not be invented. Fees match Trader's two legs.
    """
    columns = ["signal_date", "execution_date", "ticker", "leg", "side",
               "notional", "fee", "factor_weight"]
    if positions.empty:
        return pd.DataFrame(columns=columns)
    if positions.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate simulated position")
    lookup = frame.set_index(["date", "ticker"])[["execution_date", "target_open_close"]]
    if not lookup.index.is_unique:
        raise ValueError("Duplicate execution label")
    rows = []
    nav = initial_cash
    held = {(pd.Timestamp(row.date), row.ticker): row.weight for row in positions.itertuples()}
    for date, day in report.iterrows():
        date = pd.Timestamp(date)
        weights = {ticker: weight for (signal_date, ticker), weight in held.items() if signal_date == date}
        if len(weights) != int(day["selected_assets"]):
            raise ValueError("Position count differs from daily ledger")
        entry = exit_ = fee = gross = 0.0
        for ticker, weight in sorted(weights.items()):
            execution_date, ret = lookup.loc[(date, ticker)]
            if pd.isna(execution_date) or pd.Timestamp(execution_date) != pd.Timestamp(day["execution_date"]) or not np.isfinite(ret) or ret < -1:
                raise ValueError("Selected asset has no valid next-session execution return")
            open_notional = nav * abs(weight)
            close_notional = open_notional * (1 + ret)
            for leg, side, amount in (("open", "buy" if weight > 0 else "sell_short", open_notional),
                                      ("close", "sell" if weight > 0 else "buy_to_cover", close_notional)):
                rows.append({"signal_date": date, "execution_date": execution_date,
                             "ticker": ticker, "leg": leg, "side": side,
                             "notional": amount, "fee": amount * cost_rate, "factor_weight": weight})
            entry += open_notional
            exit_ += close_notional
            fee += cost_rate * (open_notional + close_notional)
            gross += nav * weight * ret
        if not np.isclose(entry + exit_, nav * day["traded_notional"], atol=1e-6) or not np.isclose(fee, nav * day["transaction_cost"], atol=1e-6):
            raise ValueError("Order turnover/fees do not reconcile with Trader")
        if not np.isclose((gross - fee) / nav, day["net_return"], atol=1e-10):
            raise ValueError("Order P&L does not reconcile with Trader")
        nav *= 1 + day["net_return"]
    if len(rows) != len(positions) * 2:
        raise ValueError("Unmatched simulated orders")
    return pd.DataFrame(rows, columns=columns)


def demo_gate(record: dict) -> bool:
    """A weak *exploratory* directional gate, not Algorithm 1 admission."""
    direction = record.get("direction_from_train")
    if direction not in (-1, 1):
        return False
    for split in ("train", "validation"):
        metrics = record.get("horizons", {}).get("1", {}).get(split, {})
        try:
            hit = metrics["rank_ic_hit_ratio"] if direction == 1 else 1.0 - metrics["rank_ic_hit_ratio"]
            values = [direction * metrics[name] for name in ("mean_daily_ic", "mean_daily_rank_ic", "icir", "rank_icir")]
            if (not all(np.isfinite(value) and value > 0 for value in values)
                    or not np.isfinite(hit) or hit <= 0.5
                    or not np.isfinite(metrics["coverage"]) or metrics["coverage"] <= 0.90
                    or not np.isfinite(metrics["turnover"]) or metrics["turnover"] >= 0.40):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def admitted_factors(miner_dir: Path, cutoff: str) -> tuple[list[dict], list[dict]]:
    registry = json.loads((miner_dir / "registry.json").read_text())
    if registry.get("as_of") != cutoff or registry.get("generator", "").split(":")[0] not in ("llm_cluster", "llm_langgraph"):
        raise ValueError("Expected an LLM-generated Miner result at the committed admission cutoff")
    if registry.get("accepted_count") != 0:
        raise ValueError("This demo expects a strict Miner failure; do not override real strict results")
    metrics = json.loads((miner_dir / "candidate_metrics.json").read_text())
    candidates = {row["candidate_id"]: candidate_from_dict(row) for row in json.loads((miner_dir / "candidates.json").read_text())}
    accepted = []
    decisions = []
    for row in metrics:
        candidate = candidates.get(row["candidate_id"])
        if candidate is None:
            raise ValueError("Miner metric does not have a frozen candidate definition")
        passed = demo_gate(row)
        decisions.append({"candidate_id": candidate.candidate_id, "demo_gate_passed": passed,
                          "strict_gate_results": row["gate_results"],
                          "one_day": row["horizons"]["1"]})
        if not passed:
            continue
        record = {
            "candidate_id": candidate.candidate_id, "op": candidate.op,
            "expression_spec": candidate.expression_spec, "expression": candidate.expression,
            "features": list(candidate.features), "hypothesis": candidate.hypothesis,
            "direction": row["direction_from_train"], "admission_policy": ADMISSION_RULE,
            "strict_paper_accepted": False, "status": "active",
            "historical_metric_cutoff": cutoff,
            # A 2026-era model generated these proposals retrospectively. Do NOT
            # forge an admission timestamp or an apparent PIT snapshot in 2022.
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        accepted.append(record)
    if not accepted:
        raise ValueError("No LLM-proposed factor passed even the exploratory execution gate")
    return accepted, decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miner-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--admission-as-of", default="2022-12-30")
    parser.add_argument("--selection-start", default="2023-01-03")
    parser.add_argument("--selection-end", default="2023-12-29")
    args = parser.parse_args()
    args.miner_dir, args.input, args.calendar = (path.resolve(strict=True) for path in (args.miner_dir, args.input, args.calendar))
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Output directory must be new or empty")
    if pd.Timestamp(args.admission_as_of) >= pd.Timestamp(args.selection_start) or pd.Timestamp(args.selection_start) > pd.Timestamp(args.selection_end):
        parser.error("Admission must precede all demo signal dates")
    sources = [
        args.input, args.calendar, HERE / "miner_lite.py", HERE / "alphax/factor_dsl.py",
        HERE / "screener_lite.py", HERE / "trader_lite.py", HERE / "run_agent_execution_demo.py",
    ]
    sources += [args.miner_dir / name for name in ("registry.json", "candidate_metrics.json", "candidates.json", "llm_candidate_audit.json", "temporal_audit.json")]
    pinned = {str(path): file_fingerprint(path) for path in sources}
    accepted, decisions = admitted_factors(args.miner_dir, args.admission_as_of)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_type": "exploratory_agent_execution_demo_not_paper_or_live_broker",
        "admission_rule": ADMISSION_RULE, "admission_cutoff": args.admission_as_of,
        "signal_dates": [args.selection_start, args.selection_end], "config": CONFIG.__dict__,
        "cost_rate_per_leg": COST_RATE, "min_suitability": 0.02, "source_fingerprints": pinned,
        "warning": "Weakened 1d admission, no paper Trader risk gate; 2023 was previously viewed; LLM model memory not certified PIT; current snapshot universe is biased; fills have notional and return only, no executable quotes or share quantities",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "admission_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    payload = {"library_version": "execution-demo-retrospective-v2", "historical_metric_cutoff": args.admission_as_of,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "admission_policy": ADMISSION_RULE, "library_scope": "retrospective_diagnostic",
               "factors": accepted, "temporal_audit": json.loads((args.miner_dir / "temporal_audit.json").read_text())}
    library_path = args.output_dir / "factor_library_demo.json"
    library_path.write_text(json.dumps(payload, indent=2) + "\n")
    screen = args.output_dir / "screener"
    command = [sys.executable, str(HERE / "screener_lite.py"), "--input", str(args.input),
               "--calendar", str(args.calendar), "--require-explicit-label-dates",
               "--allow-retrospective-library", "--factor-library", str(library_path),
               "--signal-start", args.selection_start, "--as-of", args.selection_end,
               "--output-dir", str(screen)]
    with (args.output_dir / "screener.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    registry = json.loads((screen / "registry.json").read_text())
    if registry.get("library_scope") != "retrospective_diagnostic":
        raise ValueError("Screener must label LLM historical reuse as retrospective")
    predictions = screen / "predictions.csv"
    frame = load_inputs(predictions, args.input, as_of=args.selection_end,
                        calendar_path=args.calendar, require_explicit_label_dates=True)
    report, positions = simulate(frame, CONFIG, args.selection_start, args.selection_end, COST_RATE)
    if positions.empty or not (report["selected_assets"] > 0).any():
        raise ValueError("No simulated positions; do not claim successful demo")
    selection = pd.read_csv(screen / "selection.csv")
    selected_ids = set(selection["candidate_id"])
    if not selected_ids or not selected_ids <= {item["candidate_id"] for item in accepted}:
        raise ValueError("No admitted library factor was selected by Screener")
    orders = simulated_orders(frame, report, positions, COST_RATE)
    if orders.empty or not set(orders["signal_date"]).issubset(set(pd.to_datetime(selection["date"]))):
        raise ValueError("Orders have no corresponding factor selection")
    report.to_csv(args.output_dir / "trades_daily.csv")
    positions.to_csv(args.output_dir / "positions.csv", index=False)
    orders.to_csv(args.output_dir / "simulated_orders.csv", index=False)
    for name, expected in pinned.items():
        if file_fingerprint(Path(name)) != expected:
            raise ValueError(f"Input/code changed during demo: {name}")
    summary = {"status": "completed_exploratory_demo", "library_scope": "retrospective_diagnostic",
               "generated_at_utc": payload["generated_at_utc"], "admission_rule": ADMISSION_RULE,
               "strict_paper_admitted": 0, "demo_admitted_factors": [x["candidate_id"] for x in accepted],
               "signal_rows": int(pd.read_csv(predictions).shape[0]),
               "invested_days": int((report["selected_assets"] > 0).sum()),
               "position_rows": len(positions), "simulated_order_legs": len(orders),
               "selected_factor_ids": sorted(selected_ids), "metrics": summarize(report, INITIAL_CASH, 0.0381),
               "limitations": manifest["warning"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "completed_exploratory_demo"}) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
