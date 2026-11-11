#!/usr/bin/env python3
"""Chronological, deterministic AlphaCrafter experiment runner.

Commit the schedule and policy first. Build selection-period snapshots, freeze
Trader parameters, THEN generate holdout-period snapshots and evaluate once.
No external LLM calls, retrospective-library bypass, gate overrides or resume.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from temporal import session_date, snapshot_date
from trader_lite import (
    TraderConfig, file_fingerprint, is_feasible, load_inputs, parse_configs,
    select_configuration, simulate, summarize, validate_intervals,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIGS = "0.8,0.5,50,25;0.8,0.0,50,50;0.8,1.0,50,0"


def iso(value: object) -> str:
    return str(session_date(value).date())


def build_plan(
    sessions: Sequence[object], initial_as_of: str, train_start: str,
    selection_start: str, selection_end: str, holdout_start: str, holdout_end: str,
    maintenance_days: int = 90, admission_sessions: int = 252,
    min_train_sessions: int = 252,
) -> Dict[str, object]:
    """Expanding training, rolling admission; maintenance on first session due.

    The initial requested cutoff may be a non-session: use the last observed
    session on or before it. Later rounds are >= maintenance_days calendar days
    apart, not every N rows and not anchored to a future performance measure.
    """
    for value in (initial_as_of, train_start, selection_start, selection_end, holdout_start, holdout_end):
        session_date(value)
    validate_intervals(selection_start, selection_end, holdout_start, holdout_end)
    if maintenance_days < 1 or admission_sessions < 6 or min_train_sessions < 6:
        raise ValueError("maintenance_days must be positive; admission/train need at least six sessions for 5d labels")
    calendar = pd.DatetimeIndex(sorted({session_date(value) for value in sessions}))
    if calendar.empty:
        raise ValueError("Empty feature calendar")
    if calendar[-1] < session_date(holdout_end):
        raise ValueError("Input calendar ends before holdout_end; rebuild data or explicitly shorten the experiment")
    if not session_date(train_start) <= session_date(initial_as_of) < session_date(selection_start):
        raise ValueError("Require train_start <= initial_as_of < selection_start")
    for name, start, end in (("selection", selection_start, selection_end), ("holdout", holdout_start, holdout_end)):
        if int(((calendar >= session_date(start)) & (calendar <= session_date(end))).sum()) < 2:
            raise ValueError(f"{name} requires at least two observed sessions for next-session execution")
    eligible = calendar[(calendar >= session_date(train_start)) & (calendar <= session_date(initial_as_of))]
    if len(eligible) < admission_sessions + min_train_sessions:
        raise ValueError("Insufficient pre-selection history for independent training and admission windows")
    cutoff = eligible[-1]
    rounds = []
    while cutoff <= session_date(holdout_end):
        history = calendar[(calendar >= session_date(train_start)) & (calendar <= cutoff)]
        validation = history[-admission_sessions:]
        train = history[:-admission_sessions]
        rounds.append({
            "as_of": iso(cutoff),
            "phase": "selection" if cutoff <= session_date(selection_end) else "holdout",
            "train_start": iso(train_start), "train_end": iso(train[-1]),
            "validation_start": iso(validation[0]), "validation_end": iso(cutoff),
            "observed_train_sessions": len(train), "observed_admission_sessions": len(validation),
        })
        next_index = calendar.searchsorted(cutoff + pd.Timedelta(days=maintenance_days))
        if next_index == len(calendar):
            break
        cutoff = calendar[next_index]
    return {
        "runner_version": "pit-pipeline-v1",
        "initial_as_of_requested": iso(initial_as_of),
        "initial_as_of_effective": rounds[0]["as_of"],
        "selection": {"start": iso(selection_start), "end": iso(selection_end)},
        "holdout": {"start": iso(holdout_start), "end": iso(holdout_end)},
        "maintenance_days": maintenance_days,
        "admission_sessions": admission_sessions,
        "min_train_sessions": min_train_sessions,
        "calendar_first_session": iso(calendar[0]),
        "calendar_last_session": iso(calendar[-1]),
        "rounds": rounds,
        "policy": "Expanding training + rolling admission; only matured labels; Trader frozen before holdout generation",
    }


def write_json_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def fingerprint_set(paths: Sequence[Path]) -> Dict[str, object]:
    return {str(path.resolve()): file_fingerprint(path) for path in paths}


def verify_fingerprints(fingerprints: Dict[str, object]) -> None:
    for name, expected in fingerprints.items():
        actual = file_fingerprint(Path(name))
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise ValueError(f"Pinned input/code/artifact changed during execution: {name}")


class Pipeline:
    def __init__(self, args: argparse.Namespace, plan: Dict[str, object]):
        self.args = args
        self.plan = plan
        self.root = args.output_dir.resolve()
        self.round_results: List[Dict[str, object]] = []
        self.previous_registry = None
        self.completed_artifacts: Dict[str, object] = {}
        self.stage = "initializing"

    def event(self, payload: Dict[str, object]) -> None:
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def step(self, name: str, script: str, arguments: Sequence[object]) -> None:
        self.stage = name
        command = [sys.executable, str(HERE / script), *map(str, arguments)]
        logfile = self.root / "logs" / f"{name}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.event({"event": "step_started", "name": name, "argv": command, "log": str(logfile.relative_to(self.root))})
        with logfile.open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=self.args.step_timeout)
        self.event({"event": "step_finished", "name": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {logfile}")

    def mine(self, round_plan: Dict[str, object]) -> None:
        cutoff = round_plan["as_of"]
        destination = self.root / "miner" / cutoff
        destination.mkdir(parents=True)
        # Miner --registry is read/write. Copy the predecessor into the NEW
        # round, never point it at an archived predecessor's registry itself.
        if self.previous_registry is not None:
            verify_fingerprints(self.completed_artifacts)
            shutil.copyfile(self.previous_registry, destination / "registry.json")
        next_day = iso(session_date(cutoff) + pd.Timedelta(days=1))
        arguments = [
            "--input", self.args.input, "--output-dir", destination,
            "--snapshot-dir", self.root / "snapshots", "--as-of", cutoff,
            "--train-start", round_plan["train_start"], "--train-end", round_plan["train_end"],
            "--validation-start", round_plan["validation_start"], "--validation-end", cutoff,
            # Miner diagnostics cannot evaluate this future segment at cutoff.
            "--backtest-start", next_day, "--backtest-end", next_day,
            "--maintenance-interval-days", self.args.maintenance_days,
            "--max-pairs", self.args.max_pairs, "--max-temporal", self.args.max_temporal,
        ]
        if self.args.candidate_file:
            arguments.extend(["--candidate-file", self.args.candidate_file])
        self.step(f"miner_{cutoff}", "miner_lite.py", arguments)
        snapshot = self.root / "snapshots" / f"{cutoff}.json"
        payload = json.loads(snapshot.read_text())
        if snapshot_date(payload) != session_date(cutoff):
            raise ValueError("Miner snapshot cutoff disagrees with committed schedule")
        audit = payload["temporal_audit"]
        for horizon in ("1", "5"):
            if audit["segments"][horizon]["backtest"]["eligible_rows"]:
                raise ValueError("Miner evaluated future holdout labels")
            for segment in ("train", "validation"):
                maturity = audit["segments"][horizon][segment]["last_label_end"]
                boundary = round_plan["train_end"] if segment == "train" else cutoff
                if maturity is not None and session_date(maturity) > session_date(boundary):
                    raise ValueError("Miner label maturity crossed a committed segment boundary")
        maintenance = payload["maintenance"]
        row = {
            **round_plan,
            "active_factors": len(payload["factors"]),
            "passed_current_gates": maintenance["current_round_gate_pass_count"],
            "revalidated": len(maintenance["revalidated"]),
            "pruned": len(maintenance["pruned"]),
            "snapshot_sha256": file_fingerprint(snapshot)["sha256"],
        }
        self.round_results.append(row)
        self.previous_registry = destination / "registry.json"
        self.completed_artifacts.update(fingerprint_set([snapshot, self.previous_registry]))
        self.event({"event": "snapshot_completed", **row})

    def screen(self, phase: str) -> Path:
        window = self.plan[phase]
        output = self.root / f"screener_{phase}"
        self.step(f"screener_{phase}", "screener_lite.py", [
            "--input", self.args.input, "--library-history", self.root / "snapshots",
            "--signal-start", window["start"], "--as-of", window["end"],
            "--output-dir", output,
        ])
        registry = json.loads((output / "registry.json").read_text())
        if registry["library_scope"] != "point_in_time_snapshots":
            raise ValueError("Retrospective predictions cannot enter the pipeline")
        return output / "predictions.csv"

    def freeze_selection(self, predictions: Path) -> Path:
        self.stage = "trader_selection"
        window = self.plan["selection"]
        frame = load_inputs(predictions, self.args.input, as_of=window["end"])
        results = []
        destination = self.root / "trader"
        destination.mkdir()
        for index, config in enumerate(parse_configs(self.args.configs), 1):
            report, _ = simulate(frame, config, window["start"], window["end"], self.args.cost_rate)
            report.to_csv(destination / f"selection_trial_{index}.csv")
            metrics = summarize(report, self.args.initial_cash, self.args.annual_risk_free_rate)
            results.append({"config": asdict(config), "name": config.name, **metrics})
        selected = select_configuration(results)
        path = destination / "selection.json"
        frozen = {
            "frozen_as_of": window["end"], "selection_window": window,
            "status": "selected" if selected else "no_feasible_strategy",
            "selected_config": selected["config"] if selected else None,
            "validation_results": results,
            "feasible_config_count": sum(is_feasible(result) for result in results),
            "selection_constraints": {"annualized_return": ">0.08", "sharpe": ">0.6", "max_drawdown": ">-0.08"},
            "cost_rate": self.args.cost_rate,
            "annual_risk_free_rate": self.args.annual_risk_free_rate,
            "initial_cash": self.args.initial_cash,
            "prediction_sha256": file_fingerprint(predictions)["sha256"],
            "snapshot_hashes": {row["as_of"]: row["snapshot_sha256"] for row in self.round_results},
        }
        write_json_new(path, frozen)
        self.completed_artifacts.update(fingerprint_set([path, predictions]))
        self.event({"event": "trader_frozen", "status": frozen["status"], "sha256": file_fingerprint(path)["sha256"]})
        return path

    def evaluate_holdout(self, selection: Path, predictions: Path) -> Dict[str, object]:
        self.stage = "trader_holdout"
        verify_fingerprints(self.completed_artifacts)
        # Read the committed choice; never call select_configuration on holdout.
        frozen = json.loads(selection.read_text())
        config = TraderConfig(**frozen["selected_config"]) if frozen["selected_config"] else None
        window = self.plan["holdout"]
        frame = load_inputs(predictions, self.args.input, as_of=window["end"])
        report, positions = simulate(frame, config, window["start"], window["end"], frozen["cost_rate"])
        report.to_csv(self.root / "trader" / "holdout_returns.csv")
        positions.to_csv(self.root / "trader" / "holdout_positions.csv", index=False)
        metrics = summarize(report, frozen["initial_cash"], frozen["annual_risk_free_rate"])
        return {
            "runner_version": self.plan["runner_version"],
            "status": frozen["status"], "selected_config": frozen["selected_config"],
            "selection_sha256": file_fingerprint(selection)["sha256"],
            "holdout_prediction_sha256": file_fingerprint(predictions)["sha256"],
            "maintenance_rounds": len(self.round_results),
            "holdout": metrics,
            "holdout_invested_days": int((report["selected_assets"] > 0).sum()),
            "rounds": self.round_results,
            "limitations": [
                "Deterministic templates or externally supplied fixed candidates; no historical LLM workflow",
                "Intraday Trader proxy, not the paper's persistent-holdings/margin exchange simulator",
                "Observed feature calendar, not a verified exchange calendar or point-in-time membership universe",
                "Source labels and external candidate provenance require independent audit",
                "Holdout factors adapt using already-revealed data under a precommitted policy; Trader parameters stay frozen",
            ],
        }

    def run(self) -> Dict[str, object]:
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise ValueError("Output directory must be new or empty; completed/failed runs are never overwritten")
        self.root.mkdir(parents=True, exist_ok=True)
        inputs = [self.args.input] + ([self.args.candidate_file] if self.args.candidate_file else [])
        code = [HERE / name for name in ("run_pipeline.py", "miner_lite.py", "screener_lite.py", "trader_lite.py", "temporal.py")]
        pinned = fingerprint_set(inputs + code)
        parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()}
        write_json_new(self.root / "plan.json", self.plan)
        write_json_new(self.root / "manifest.json", {
            "parameters": parameters, "fingerprints": pinned,
            "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
            "candidate_provenance": "External fixed hypotheses are user-attested, not certified point-in-time" if self.args.candidate_file else "Deterministic template generator",
        })
        try:
            for round_plan in self.plan["rounds"]:
                if round_plan["phase"] == "selection":
                    self.mine(round_plan)
            selection_predictions = self.screen("selection")
            selection = self.freeze_selection(selection_predictions)
            verify_fingerprints(pinned)
            for round_plan in self.plan["rounds"]:
                if round_plan["phase"] == "holdout":
                    self.mine(round_plan)
            holdout_predictions = self.screen("holdout")
            summary = self.evaluate_holdout(selection, holdout_predictions)
            verify_fingerprints(pinned)
            verify_fingerprints(self.completed_artifacts)
            pd.DataFrame(self.round_results).to_csv(self.root / "rounds.csv", index=False)
            write_json_new(self.root / "summary.json", summary)
            write_json_new(self.root / "status.json", {"status": "completed"})
            self.event({"event": "completed"})
            return summary
        except (Exception, KeyboardInterrupt) as exc:
            self.event({"event": "failed", "stage": self.stage, "error": f"{type(exc).__name__}: {exc}"})
            write_json_new(self.root / "status.json", {"status": "failed", "stage": self.stage, "error": f"{type(exc).__name__}: {exc}"})
            raise


def main() -> None:
    base = HERE.parent / "machine-learn"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base / "data/processed/sp500_features.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, help="Fixed hypotheses decided before selection; no external LLM calls")
    parser.add_argument("--initial-as-of", default="2022-12-30")
    parser.add_argument("--train-start", default="2016-01-04")
    parser.add_argument("--selection-start", default="2023-01-03")
    parser.add_argument("--selection-end", default="2023-12-29")
    parser.add_argument("--holdout-start", default="2024-01-02")
    parser.add_argument("--holdout-end", default="2026-02-27")
    parser.add_argument("--maintenance-days", type=int, default=90)
    parser.add_argument("--admission-sessions", type=int, default=252)
    parser.add_argument("--min-train-sessions", type=int, default=252)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--max-temporal", type=int, default=20)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--cost-rate", type=float, default=.0001)
    parser.add_argument("--annual-risk-free-rate", type=float, default=.0381)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.)
    parser.add_argument("--step-timeout", type=float, default=None, help="Optional subprocess timeout in seconds; failures preserve logs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the schedule without creating output or running agents")
    args = parser.parse_args()
    try:
        args.input = args.input.resolve(strict=True)
        if args.candidate_file:
            args.candidate_file = args.candidate_file.resolve(strict=True)
        parse_configs(args.configs)
        if not np.isfinite(args.cost_rate) or args.cost_rate < 0:
            raise ValueError("cost-rate must be finite and non-negative")
        if not np.isfinite(args.initial_cash) or args.initial_cash <= 0:
            raise ValueError("initial-cash must be finite and positive")
        if not np.isfinite(args.annual_risk_free_rate) or args.annual_risk_free_rate <= -1:
            raise ValueError("annual-risk-free-rate must be finite and greater than -1")
        if args.max_pairs < 0 or args.max_temporal < 0:
            raise ValueError("Template candidate limits cannot be negative")
        if args.step_timeout is not None and (not np.isfinite(args.step_timeout) or args.step_timeout <= 0):
            raise ValueError("step-timeout must be finite and positive")
        calendar = pd.read_csv(args.input, usecols=["date"], parse_dates=["date"])["date"].drop_duplicates()
        plan = build_plan(calendar, args.initial_as_of, args.train_start, args.selection_start, args.selection_end,
                          args.holdout_start, args.holdout_end, args.maintenance_days, args.admission_sessions, args.min_train_sessions)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        summary = Pipeline(args, plan).run()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
    print(json.dumps({key: summary[key] for key in ("status", "selected_config", "maintenance_rounds", "holdout")}, indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
