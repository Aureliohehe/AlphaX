#!/usr/bin/env python3
"""Chronological, deterministic AlphaCrafter experiment runner.

Commit the schedule and policy first. Build selection-period snapshots, freeze
Trader parameters, THEN generate holdout-period snapshots and evaluate once.
Committed stages can be resumed through the end of holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from alphax.checkpoint import lock_run, publish_json
from alphax.experiment import artifact_inventory, build_manifest
from temporal import (
    LABEL_END_COLUMNS, calendar_audit, label_end_source, read_calendar, session_date, snapshot_date,
    validate_previous_registry,
)
from trader_lite import (
    TraderConfig, file_fingerprint, is_feasible, load_inputs, parse_configs,
    select_configuration, simulate, summarize, validate_intervals,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIGS = "0.8,0.5,50,25;0.8,0.0,50,50;0.8,1.0,50,0"
CHECKPOINT_VERSION = 3
SCREENER_SELECTION = "screener_selection"
TRADER_SELECTION = "trader_selection"
SCREENER_HOLDOUT = "screener_holdout"
TRADER_HOLDOUT = "trader_holdout"
HOLDOUT_EVALUATION = "holdout_evaluation"
HOLDOUT_EXPORTS = {
    "holdout_returns.csv": "trader/holdout_returns.csv",
    "holdout_positions.csv": "trader/holdout_positions.csv",
    "rounds.csv": "rounds.csv",
    "summary.json": "summary.json",
}


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


def input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {name: path for name, path in {
        "panel": args.input, "calendar": args.calendar, "candidate_file": args.candidate_file,
    }.items() if path is not None}


def normalized_config(args: argparse.Namespace) -> dict:
    config = {key: value for key, value in vars(args).items()
              if key not in {"input", "calendar", "candidate_file", "output_dir", "dry_run", "resume"}}
    for key in ("initial_as_of", "train_start", "selection_start", "selection_end", "holdout_start", "holdout_end"):
        config[key] = iso(config[key])
    config["configs"] = [asdict(trader_config) for trader_config in parse_configs(args.configs)]
    return config


class Pipeline:
    def __init__(self, args: argparse.Namespace, plan: Dict[str, object]):
        self.args = args
        self.plan = plan
        self.root = args.output_dir.resolve()
        self.checkpoints = self.root / "checkpoints"
        self.holdout_boundary = self.checkpoints / "holdout_started.json"
        self.log_dir = self.root / "logs"
        self.lock_fd = None
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
        logfile = self.log_dir / f"{name}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self.event({"event": "step_started", "name": name, "argv": command, "log": str(logfile.relative_to(self.root))})
        with logfile.open("x") as log:
            descriptors = () if self.lock_fd is None else (self.lock_fd,)
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=self.args.step_timeout, pass_fds=descriptors,
            )
        self.event({"event": "step_finished", "name": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {logfile}")

    def mine(self, round_plan: Dict[str, object]) -> None:
        cutoff = round_plan["as_of"]
        self.stage = f"miner_{cutoff}"
        destination = self.root / "miner" / cutoff
        snapshot_dir = self.root / "snapshots"
        staging = self.root / ".staging" / cutoff
        output = staging / "miner"
        snapshots = staging / "snapshots"
        output.mkdir(parents=True)
        # Miner --registry is read/write. Copy the predecessor into the NEW
        # round, never point it at an archived predecessor's registry itself.
        if self.previous_registry is not None:
            verify_fingerprints(self.completed_artifacts)
            shutil.copyfile(self.previous_registry, output / "registry.json")
        next_day = iso(session_date(cutoff) + pd.Timedelta(days=1))
        arguments = [
            "--input", self.args.input, "--output-dir", output,
            "--snapshot-dir", snapshots, "--as-of", cutoff,
            "--train-start", round_plan["train_start"], "--train-end", round_plan["train_end"],
            "--validation-start", round_plan["validation_start"], "--validation-end", cutoff,
            # Miner diagnostics cannot evaluate this future segment at cutoff.
            "--backtest-start", next_day, "--backtest-end", next_day,
            "--maintenance-interval-days", self.args.maintenance_days,
            "--max-pairs", self.args.max_pairs, "--max-temporal", self.args.max_temporal,
        ]
        if self.args.candidate_file:
            arguments.extend(["--candidate-file", self.args.candidate_file])
        if self.args.require_explicit_label_dates:
            arguments.append("--require-explicit-label-dates")
        if self.args.calendar:
            arguments.extend(["--calendar", self.args.calendar])
        self.step(self.stage, "miner_lite.py", arguments)
        snapshot = snapshots / f"{cutoff}.json"
        registry = output / "registry.json"
        row = self.round_record(round_plan, snapshot, registry)
        published_snapshot = snapshot_dir / snapshot.name
        # Preserve the Miner registry's existing public snapshot path.
        payload = json.loads(registry.read_text())
        payload["snapshot_file"] = str(published_snapshot)
        registry.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
        for target in (destination, published_snapshot):
            if target.exists() or target.is_symlink():
                raise ValueError(f"Cannot publish over existing Miner output: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
        output.rename(destination)
        snapshot.rename(published_snapshot)
        snapshot, registry = published_snapshot, destination / "registry.json"
        checkpoint = self.checkpoints / f"{cutoff}.json"
        publish_json(checkpoint, {"version": CHECKPOINT_VERSION, "round": row,
                                  "files": self.relative_fingerprints([snapshot, registry])})
        self.completed_artifacts.update(fingerprint_set([checkpoint]))
        self.register_round(row, snapshot, registry)
        # The snapshot subdirectory and staging directory are empty after
        # publication. A crash here leaves only harmless empty directories.
        snapshots.rmdir()
        staging.rmdir()
        self.event({"event": "snapshot_completed", **row})

    def round_record(self, round_plan: dict, snapshot: Path, registry: Path) -> dict:
        cutoff = round_plan["as_of"]
        payload = json.loads(snapshot.read_text())
        if snapshot_date(payload) != session_date(cutoff):
            raise ValueError("Miner snapshot cutoff disagrees with committed schedule")
        registry_payload = json.loads(registry.read_text())
        validate_previous_registry(registry_payload, cutoff)
        if session_date(registry_payload["as_of"]) != session_date(cutoff):
            raise ValueError("Miner registry cutoff disagrees with committed schedule")
        expected_history = [row["as_of"] for row in self.round_results] + [cutoff]
        if [entry["as_of"] for entry in registry_payload["run_history"]] != expected_history:
            raise ValueError("Miner registry history disagrees with committed schedule")
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
        return {
            **round_plan,
            "active_factors": len(payload["factors"]),
            "passed_current_gates": maintenance["current_round_gate_pass_count"],
            "revalidated": len(maintenance["revalidated"]),
            "pruned": len(maintenance["pruned"]),
            "snapshot_sha256": file_fingerprint(snapshot)["sha256"],
        }

    def register_round(self, row: dict, snapshot: Path, registry: Path) -> None:
        self.round_results.append(row)
        self.previous_registry = registry
        self.completed_artifacts.update(fingerprint_set([snapshot, registry]))

    def relative_fingerprints(self, paths: Sequence[Path]) -> dict:
        return {path.relative_to(self.root).as_posix(): {
            key: value for key, value in file_fingerprint(path).items() if key != "path"
        } for path in paths}

    def restore(self, manifest: dict, selection_rounds: list[dict]) -> tuple[Path | None, Path | None, Path | None, dict | None]:
        status = self.root / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") != "failed":
            raise ValueError("Resume requires an interrupted or failed run; completed runs cannot be reopened")
        initial = self.checkpoints / "run.json"
        if not initial.is_file():
            raise ValueError("No initialized Miner checkpoints in this run")
        manifest_path, plan_path = self.root / "manifest.json", self.root / "plan.json"
        original = json.loads(manifest_path.read_text())
        if json.loads(initial.read_text()) != {
            "version": CHECKPOINT_VERSION, "run_id": original["run_id"],
            "files": self.relative_fingerprints([manifest_path, plan_path]),
        }:
            raise ValueError("Resume manifest or plan no longer matches its initialization checkpoint")
        if (original["experiment_fingerprint"] != manifest["experiment_fingerprint"]
                or original["fingerprints"] != manifest["fingerprints"]
                or json.loads(plan_path.read_text()) != self.plan):
            raise ValueError("Resume inputs, configuration, code, runtime, or plan changed")
        self.completed_artifacts.update(fingerprint_set([manifest_path, plan_path, initial]))

        selection_marker = self.checkpoints / "selection_closed.json"
        holdout_marker = self.holdout_boundary
        holdout_started = holdout_marker.exists()
        planned = self.plan["rounds"] if holdout_started else selection_rounds
        checkpoint_names = {path.name for path in self.checkpoints.glob("*.json")} - {
            initial.name, selection_marker.name, holdout_marker.name,
        }
        committed = planned[:len(checkpoint_names)]
        if checkpoint_names != {f"{row['as_of']}.json" for row in committed}:
            raise ValueError("Miner checkpoints must be a contiguous prefix of the committed plan")

        self.restore_miner_rounds(committed[:len(selection_rounds)])
        predictions = selection = holdout_predictions = summary = None
        if selection_marker.exists():
            if (len(self.round_results) != len(selection_rounds)
                    or json.loads(selection_marker.read_text()) != self.selection_started_record()):
                raise ValueError("Selection Screener marker disagrees with the committed Miner rounds")
            self.completed_artifacts.update(fingerprint_set([selection_marker]))
            predictions, selection = self.restore_selection_outputs()
        elif any((self.root / name).exists() for name in (SCREENER_SELECTION, "trader")):
            raise ValueError("Selection output exists without the selection start marker")
        if holdout_started:
            if selection is None or json.loads(holdout_marker.read_text()) != self.holdout_started_record(selection):
                raise ValueError("Holdout start marker disagrees with the frozen Trader checkpoint")
            self.completed_artifacts.update(fingerprint_set([holdout_marker]))
            self.restore_miner_rounds(committed[len(selection_rounds):])
            if len(committed) == len(planned):
                holdout_predictions, summary = self.restore_holdout_outputs(selection)
            else:
                self.reject_unexpected_holdout_outputs()
        else:
            self.reject_unexpected_holdout_outputs()

        completed_dates = {row["as_of"] for row in self.round_results}
        allowed_dates = completed_dates.copy()
        partial = []
        if len(committed) < len(planned):
            cutoff = planned[len(committed)]["as_of"]
            stage = self.root / ".staging" / cutoff
            if stage.is_dir() and not stage.is_symlink():
                allowed_dates.add(cutoff)
                partial = [self.root / "miner" / cutoff, self.root / "snapshots" / f"{cutoff}.json", stage]
        self.check_resume_archive(completed_dates, allowed_dates)
        allowed_stages = allowed_dates.copy()
        if holdout_started and len(committed) == len(planned) and summary is None:
            pending = TRADER_HOLDOUT if holdout_predictions is not None else SCREENER_HOLDOUT
        elif selection_marker.exists() and selection is None:
            pending = TRADER_SELECTION if predictions is not None else SCREENER_SELECTION
        else:
            pending = None
        if pending:
            allowed_stages.add(pending)
            partial.append(self.root / ".staging" / pending)
        staged = self.root / ".staging"
        if staged.exists() and not {path.name for path in staged.iterdir()} <= allowed_stages:
            raise ValueError("Unexpected staged rounds or phase outputs for resume")

        attempts = self.root / "attempts"
        attempts.mkdir(exist_ok=True)
        archive = Path(mkdtemp(prefix="resume-", dir=attempts))
        # Move the staging directory last. If this cleanup is itself
        # interrupted, it still identifies any remaining partial publication.
        for path in [status, self.root / "artifacts.json", *partial]:
            if path.exists() or path.is_symlink():
                destination = archive / path.relative_to(self.root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
        self.log_dir = self.root / "logs" / archive.name
        self.event({"event": "resumed", "run_id": original["run_id"],
                    "completed_rounds": [row["as_of"] for row in self.round_results],
                    "screener_restored": predictions is not None, "trader_frozen_restored": selection is not None,
                    "holdout_screener_restored": holdout_predictions is not None,
                    "holdout_evaluation_restored": summary is not None,
                    "previous_attempt": str(archive.relative_to(self.root))})
        return predictions, selection, holdout_predictions, summary

    def restore_miner_rounds(self, rounds: list[dict]) -> None:
        for round_plan in rounds:
            cutoff = round_plan["as_of"]
            checkpoint = self.checkpoints / f"{cutoff}.json"
            snapshot = self.root / "snapshots" / checkpoint.name
            registry = self.root / "miner" / cutoff / "registry.json"
            record = json.loads(checkpoint.read_text())
            if record.get("version") != CHECKPOINT_VERSION or record.get("files") != self.relative_fingerprints([snapshot, registry]):
                raise ValueError(f"Committed Miner snapshot or registry changed for {cutoff}")
            row = self.round_record(round_plan, snapshot, registry)
            if row != record.get("round"):
                raise ValueError(f"Miner checkpoint disagrees with the committed round {cutoff}")
            self.completed_artifacts.update(fingerprint_set([checkpoint]))
            self.register_round(row, snapshot, registry)

    def snapshot_hashes(self) -> dict:
        return {row["as_of"]: row["snapshot_sha256"] for row in self.round_results}

    def selection_started_record(self) -> dict:
        return {"version": CHECKPOINT_VERSION, "stage": SCREENER_SELECTION, "snapshot_hashes": self.snapshot_hashes()}

    def holdout_started_record(self, selection: Path) -> dict:
        return {"version": CHECKPOINT_VERSION, "stage": "holdout",
                "trader_checkpoint_sha256": file_fingerprint(selection.with_name("checkpoint.json"))["sha256"]}

    def stage_record(self, stage: str, output: Path, inputs: dict, *, ignore: tuple[str, ...] = ()) -> dict:
        files = {}
        for path in sorted(output.iterdir()):
            if path.name == "checkpoint.json" or path.name in ignore:
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Unexpected file in {stage} output: {path.name}")
            fingerprint = file_fingerprint(path)
            files[path.name] = {"sha256": fingerprint["sha256"], "bytes": fingerprint["bytes"]}
        return {"version": CHECKPOINT_VERSION, "stage": stage, "inputs": inputs, "files": files}

    def publish_stage(self, stage: str, output: Path, destination: Path, inputs: dict) -> None:
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Cannot publish over an existing stage: {destination}")
        record = self.stage_record(stage, output, inputs)
        publish_json(output / "checkpoint.json", record)
        # The directory rename publishes the stage files and checkpoint together.
        output.rename(destination)
        self.pin_stage(destination, record)

    def pin_stage(self, directory: Path, record: dict) -> None:
        self.completed_artifacts.update(fingerprint_set([
            directory / name for name in (*record["files"], "checkpoint.json")
        ]))

    def verify_stage(self, stage: str, directory: Path, inputs: dict, *, ignore: tuple[str, ...] = ()) -> None:
        checkpoint = directory / "checkpoint.json"
        if directory.is_symlink() or checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f"Missing committed {stage} checkpoint")
        record = self.stage_record(stage, directory, inputs, ignore=ignore)
        if json.loads(checkpoint.read_text()) != record:
            raise ValueError(f"Committed {stage} output or inputs changed")
        self.pin_stage(directory, record)

    def trader_inputs(self, predictions: Path) -> dict:
        return {
            "prediction_sha256": file_fingerprint(predictions)["sha256"],
            "snapshot_hashes": self.snapshot_hashes(),
            "screener_checkpoint_sha256": file_fingerprint(predictions.with_name("checkpoint.json"))["sha256"],
        }

    def restore_selection_outputs(self) -> tuple[Path | None, Path | None]:
        screener = self.root / SCREENER_SELECTION
        trader = self.root / "trader"
        if not screener.exists():
            if trader.exists():
                raise ValueError("Trader output requires the committed selection Screener")
            return None, None
        self.verify_stage(SCREENER_SELECTION, screener, {"snapshot_hashes": self.snapshot_hashes()})
        self.validate_screen_registry(screener, "selection")
        predictions = screener / "predictions.csv"
        if not trader.exists():
            return predictions, None
        inputs = self.trader_inputs(predictions)
        self.verify_stage(TRADER_SELECTION, trader, inputs, ignore=("holdout_returns.csv", "holdout_positions.csv"))
        selection = trader / "selection.json"
        frozen = json.loads(selection.read_text())
        expected = {
            "selection_window": self.plan["selection"], "frozen_as_of": self.plan["selection"]["end"],
            "prediction_sha256": inputs["prediction_sha256"], "snapshot_hashes": inputs["snapshot_hashes"],
        }
        if any(frozen.get(key) != value for key, value in expected.items()) or frozen.get("status") != (
            "selected" if frozen.get("selected_config") is not None else "no_feasible_strategy"
        ):
            raise ValueError("Frozen Trader decision disagrees with the selection inputs")
        return predictions, selection

    def reject_unexpected_holdout_outputs(self) -> None:
        for name in (SCREENER_HOLDOUT, HOLDOUT_EVALUATION, *HOLDOUT_EXPORTS.values()):
            path = self.root / name
            if path.exists() or path.is_symlink():
                raise ValueError(f"Unexpected {name} before its holdout prerequisites were committed")

    def restore_holdout_outputs(self, selection: Path) -> tuple[Path | None, dict | None]:
        screener = self.root / SCREENER_HOLDOUT
        evaluation = self.root / HOLDOUT_EVALUATION
        if not screener.exists() and not screener.is_symlink():
            self.reject_unexpected_holdout_outputs()
            return None, None
        self.verify_stage(SCREENER_HOLDOUT, screener, self.screener_inputs("holdout"))
        self.validate_screen_registry(screener, "holdout")
        predictions = screener / "predictions.csv"
        if not evaluation.exists() and not evaluation.is_symlink():
            if any((self.root / name).exists() or (self.root / name).is_symlink()
                   for name in HOLDOUT_EXPORTS.values()):
                raise ValueError("Published holdout results require a committed evaluation")
            return predictions, None
        inputs = self.holdout_evaluation_inputs(selection, predictions)
        self.verify_stage(TRADER_HOLDOUT, evaluation, inputs)
        frozen = json.loads(selection.read_text())
        summary = json.loads((evaluation / "summary.json").read_text())
        expected = {
            "status": frozen["status"], "selected_config": frozen["selected_config"],
            "selection_sha256": inputs["selection_sha256"],
            "holdout_prediction_sha256": inputs["prediction_sha256"], "rounds": self.round_results,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("Committed holdout summary disagrees with the frozen decision or inputs")
        self.verify_holdout_exports()
        return predictions, summary

    def check_resume_archive(self, completed: set[str], allowed: set[str]) -> None:
        for folder, suffix in (("miner", ""), ("snapshots", ".json")):
            archive = self.root / folder
            actual = {path.name for path in archive.iterdir()} if archive.exists() else set()
            if not {day + suffix for day in completed} <= actual <= {day + suffix for day in allowed}:
                raise ValueError(f"Unexpected {folder} archive contents for Miner resume")

    def verify_archive(self, phase: str) -> None:
        """Reject extra snapshots, not just modifications of known files."""
        expected = {f"{row['as_of']}.json" for row in self.round_results}
        actual = {path.name for path in (self.root / "snapshots").glob("*.json")}
        if actual != expected:
            raise ValueError("Snapshot archive differs from the committed/completed schedule")
        if any(session_date(row["as_of"]) > session_date(self.plan[phase]["end"]) for row in self.round_results):
            raise ValueError("Archive contains a snapshot beyond this phase cutoff")
        verify_fingerprints(self.completed_artifacts)

    def screen(self, phase: str) -> Path:
        self.stage = f"screener_{phase}"
        self.verify_archive(phase)
        window = self.plan[phase]
        destination = self.root / self.stage
        output = self.root / ".staging" / self.stage
        output.mkdir(parents=True)
        arguments = [
            "--input", self.args.input, "--library-history", self.root / "snapshots",
            "--signal-start", window["start"], "--as-of", window["end"],
            "--output-dir", output,
        ]
        if self.args.require_explicit_label_dates:
            arguments.append("--require-explicit-label-dates")
        if self.args.calendar:
            arguments.extend(["--calendar", self.args.calendar])
        self.step(self.stage, "screener_lite.py", arguments)
        self.validate_screen_registry(output, phase)
        predictions = output / "predictions.csv"
        self.verify_archive(phase)
        if not predictions.is_file():
            raise ValueError(f"{self.stage} did not write predictions.csv")
        self.publish_stage(self.stage, output, destination, self.screener_inputs(phase))
        return destination / predictions.name

    def screener_inputs(self, phase: str) -> dict:
        inputs = {"snapshot_hashes": self.snapshot_hashes()}
        if phase == "holdout":
            inputs["holdout_started_sha256"] = file_fingerprint(self.holdout_boundary)["sha256"]
        return inputs

    def validate_screen_registry(self, output: Path, phase: str) -> None:
        window = self.plan[phase]
        registry = json.loads((output / "registry.json").read_text())
        if registry["library_scope"] != "point_in_time_snapshots":
            raise ValueError("Retrospective predictions cannot enter the pipeline")
        if registry["as_of"] != window["end"] or registry["signal_start"] != window["start"]:
            raise ValueError("Screener window differs from the committed schedule")

    def freeze_selection(self, predictions: Path) -> Path:
        self.stage = TRADER_SELECTION
        verify_fingerprints(self.completed_artifacts)
        inputs = self.trader_inputs(predictions)
        window = self.plan["selection"]
        frame = load_inputs(predictions, self.args.input, as_of=window["end"], calendar_path=self.args.calendar,
                            require_explicit_label_dates=self.args.require_explicit_label_dates)
        results = []
        destination = self.root / "trader"
        output = self.root / ".staging" / TRADER_SELECTION
        output.mkdir(parents=True)
        for index, config in enumerate(parse_configs(self.args.configs), 1):
            report, _ = simulate(frame, config, window["start"], window["end"], self.args.cost_rate)
            report.to_csv(output / f"selection_trial_{index}.csv")
            metrics = summarize(report, self.args.initial_cash, self.args.annual_risk_free_rate)
            results.append({"config": asdict(config), "name": config.name, **metrics})
        selected = select_configuration(results)
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
            "label_end_source": label_end_source(frame, 1),
            "calendar_sha256": file_fingerprint(self.args.calendar)["sha256"] if self.args.calendar else None,
            "prediction_sha256": inputs["prediction_sha256"],
            "snapshot_hashes": inputs["snapshot_hashes"],
        }
        write_json_new(output / "selection.json", frozen)
        verify_fingerprints(self.completed_artifacts)
        self.publish_stage(TRADER_SELECTION, output, destination, inputs)
        path = destination / "selection.json"
        self.event({"event": "trader_frozen", "status": frozen["status"], "sha256": file_fingerprint(path)["sha256"]})
        return path

    def holdout_evaluation_inputs(self, selection: Path, predictions: Path) -> dict:
        return {
            "selection_sha256": file_fingerprint(selection)["sha256"],
            "prediction_sha256": file_fingerprint(predictions)["sha256"],
            "screener_checkpoint_sha256": file_fingerprint(predictions.with_name("checkpoint.json"))["sha256"],
            "holdout_started_sha256": file_fingerprint(self.holdout_boundary)["sha256"],
        }

    def evaluate_holdout(self, selection: Path, predictions: Path) -> Dict[str, object]:
        self.stage = TRADER_HOLDOUT
        verify_fingerprints(self.completed_artifacts)
        inputs = self.holdout_evaluation_inputs(selection, predictions)
        # Read the committed choice; never call select_configuration on holdout.
        frozen = json.loads(selection.read_text())
        config = TraderConfig(**frozen["selected_config"]) if frozen["selected_config"] else None
        window = self.plan["holdout"]
        frame = load_inputs(predictions, self.args.input, as_of=window["end"], calendar_path=self.args.calendar,
                            require_explicit_label_dates=self.args.require_explicit_label_dates)
        report, positions = simulate(frame, config, window["start"], window["end"], frozen["cost_rate"])
        output = self.root / ".staging" / TRADER_HOLDOUT
        output.mkdir(parents=True)
        report.to_csv(output / "holdout_returns.csv")
        positions.to_csv(output / "holdout_positions.csv", index=False)
        metrics = summarize(report, frozen["initial_cash"], frozen["annual_risk_free_rate"])
        summary = {
            "runner_version": self.plan["runner_version"],
            "status": frozen["status"], "selected_config": frozen["selected_config"],
            "selection_sha256": inputs["selection_sha256"],
            "holdout_prediction_sha256": inputs["prediction_sha256"],
            "maintenance_rounds": len(self.round_results),
            "label_end_source": label_end_source(frame, 1),
            "calendar_audit": self.plan["calendar_audit"],
            "holdout": metrics,
            "holdout_invested_days": int((report["selected_assets"] > 0).sum()),
            "rounds": self.round_results,
            "limitations": [
                "Deterministic templates or externally supplied fixed candidates; no historical LLM workflow",
                "Intraday Trader proxy, not the paper's persistent-holdings/margin exchange simulator",
                "Calendar identity/authority and point-in-time constituent membership still require independent audit",
                "Source labels and external candidate provenance require independent audit",
                "Holdout factors adapt using already-revealed data under a precommitted policy; Trader parameters stay frozen",
            ],
        }
        pd.DataFrame(self.round_results).to_csv(output / "rounds.csv", index=False)
        write_json_new(output / "summary.json", summary)
        verify_fingerprints(self.completed_artifacts)
        self.publish_stage(TRADER_HOLDOUT, output, self.root / HOLDOUT_EVALUATION, inputs)
        return summary

    def verify_holdout_exports(self) -> None:
        for name, relative in HOLDOUT_EXPORTS.items():
            source = self.root / HOLDOUT_EVALUATION / name
            target = self.root / relative
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"Invalid published holdout result: {relative}")
            if target.exists() and not target.samefile(source):
                expected = self.completed_artifacts[str(source.resolve())]
                actual = file_fingerprint(target)
                if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                    raise ValueError(f"Published holdout result changed: {relative}")

    def finalize(self, pinned: dict) -> None:
        self.stage = "finalizing"
        verify_fingerprints(pinned)
        verify_fingerprints(self.completed_artifacts)
        self.verify_holdout_exports()
        for name, relative in HOLDOUT_EXPORTS.items():
            target = self.root / relative
            if not target.exists():
                os.link(self.root / HOLDOUT_EVALUATION / name, target)
        publish_json(self.root / "artifacts.json", artifact_inventory(self.root))
        events = self.root / "events.jsonl"
        if json.dumps({"event": "completed"}) not in events.read_text().splitlines():
            self.event({"event": "completed"})
        # Consumers may read a completed run only after the results and index exist.
        publish_json(self.root / "status.json", {"status": "completed"})

    def run(self, pinned_inputs: dict | None = None) -> Dict[str, object]:
        resume = getattr(self.args, "resume", False)
        if resume:
            if not self.root.is_dir():
                raise ValueError("Resume requires an existing output directory")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        with lock_run(self.root) as fd:
            self.lock_fd = fd
            try:
                if not resume and any(self.root.iterdir()):
                    raise ValueError("Output directory must be new or empty; existing runs are never overwritten without --resume")
                return self.run_locked(pinned_inputs, resume)
            finally:
                self.lock_fd = None

    def run_locked(self, pinned_inputs: dict | None, resume: bool) -> dict:
        checkout = HERE.parents[1]
        inputs = input_paths(self.args)
        if pinned_inputs is None:
            pinned_inputs = fingerprint_set(list(inputs.values()))
        code = [HERE / name for name in (
            "run_pipeline.py", "miner_lite.py", "screener_lite.py", "trader_lite.py",
            "temporal.py", "admission.py", "alphax/__init__.py", "alphax/factor_dsl.py", "alphax/experiment.py",
            "alphax/checkpoint.py",
        )]
        code.append(HERE.parent / "machine-learn/data_contract.py")
        pinned = {**pinned_inputs, **fingerprint_set(code)}
        parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()}
        manifest = build_manifest(
            parameters, normalized_config(self.args),
            {name: pinned_inputs[str(path.resolve())] for name, path in inputs.items()},
            {path.relative_to(checkout).as_posix(): pinned[str(path.resolve())] for path in code},
            pinned, self.plan, checkout,
        )
        selection_rounds = [row for row in self.plan["rounds"] if row["phase"] == "selection"]
        selection_predictions = selection = holdout_predictions = summary = None
        if resume:
            verify_fingerprints(pinned)
            selection_predictions, selection, holdout_predictions, summary = self.restore(manifest, selection_rounds)
        else:
            manifest_path, plan_path = self.root / "manifest.json", self.root / "plan.json"
            publish_json(manifest_path, manifest)
            publish_json(plan_path, self.plan)
            initial = self.checkpoints / "run.json"
            publish_json(initial, {"version": CHECKPOINT_VERSION, "run_id": manifest["run_id"],
                                   "files": self.relative_fingerprints([manifest_path, plan_path])})
            self.completed_artifacts.update(fingerprint_set([manifest_path, plan_path, initial]))
        try:
            if not self.holdout_boundary.exists():
                for round_plan in selection_rounds[len(self.round_results):]:
                    self.mine(round_plan)
                self.verify_archive("selection")
                verify_fingerprints(pinned)
                boundary = self.checkpoints / "selection_closed.json"
                if not boundary.exists():
                    publish_json(boundary, self.selection_started_record())
                    self.completed_artifacts.update(fingerprint_set([boundary]))
                if selection_predictions is None:
                    selection_predictions = self.screen("selection")
                if selection is None:
                    selection = self.freeze_selection(selection_predictions)
                verify_fingerprints(pinned)
                verify_fingerprints(self.completed_artifacts)
                publish_json(self.holdout_boundary, self.holdout_started_record(selection))
                self.completed_artifacts.update(fingerprint_set([self.holdout_boundary]))
            for round_plan in self.plan["rounds"][len(self.round_results):]:
                self.mine(round_plan)
            if holdout_predictions is None:
                holdout_predictions = self.screen("holdout")
            if summary is None:
                verify_fingerprints(pinned)
                summary = self.evaluate_holdout(selection, holdout_predictions)
            self.finalize(pinned)
            return summary
        except (Exception, KeyboardInterrupt) as exc:
            failure = {"status": "failed", "stage": self.stage, "error": f"{type(exc).__name__}: {exc}"}
            # A disk/logging failure must not mask the original stage failure
            # with a secondary FileExistsError or replace an existing status.
            try:
                self.event({"event": "failed", **failure})
            except OSError:
                pass
            if not (self.root / "artifacts.json").exists():
                try:
                    publish_json(self.root / "artifacts.json", artifact_inventory(self.root))
                except OSError:
                    pass
            if not (self.root / "status.json").exists():
                try:
                    publish_json(self.root / "status.json", failure)
                except OSError:
                    pass
            raise


def main() -> None:
    base = HERE.parent / "machine-learn"
    parser = argparse.ArgumentParser(description=__doc__)

    files_args = parser.add_argument_group("Input and output")
    files_args.add_argument("--input", type=Path, default=base / "data/processed/sp500_features.csv.gz")
    files_args.add_argument("--output-dir", type=Path, required=True)
    files_args.add_argument("--candidate-file", type=Path, help="Fixed hypotheses decided before selection; no external LLM calls")
    files_args.add_argument("--calendar", type=Path, help="Reference date CSV: require complete feature sessions across the experiment")

    windows_args = parser.add_argument_group("Research windows and data coverage")
    windows_args.add_argument("--initial-as-of", default="2022-12-30",
                              help="First factor-library cutoff (default: %(default)s)")
    windows_args.add_argument("--train-start", default="2016-01-04",
                              help="First date eligible for training (default: %(default)s)")
    windows_args.add_argument("--selection-start", default="2023-01-03",
                              help="First strategy-selection date (default: %(default)s)")
    windows_args.add_argument("--selection-end", default="2023-12-29",
                              help="Last strategy-selection date (default: %(default)s)")
    windows_args.add_argument("--holdout-start", default="2024-01-02",
                              help="First holdout date for the frozen strategy (default: %(default)s)")
    windows_args.add_argument("--holdout-end", default="2026-02-27",
                              help="Last holdout date (default: %(default)s)")
    windows_args.add_argument("--allow-feature-warmup-sessions", type=int, default=0,
                        help="Explicitly permit at most N missing reference sessions before the first feature date; internal gaps still fail")
    windows_args.add_argument("--require-explicit-label-dates", action="store_true", help="Reject legacy panels without label_end_1d/label_end_5d")

    mining_args = parser.add_argument_group("Candidate generation and maintenance")
    mining_args.add_argument("--maintenance-days", type=int, default=90,
                             help="Calendar days between maintenance rounds (default: %(default)s)")
    mining_args.add_argument("--admission-sessions", type=int, default=252,
                             help="Trading sessions in each admission window (default: %(default)s)")
    mining_args.add_argument("--min-train-sessions", type=int, default=252,
                             help="Minimum training trading sessions before admission (default: %(default)s)")
    mining_args.add_argument("--max-pairs", type=int, default=60)
    mining_args.add_argument("--max-temporal", type=int, default=20)

    trading_args = parser.add_argument_group("Strategy and transaction costs")
    trading_args.add_argument("--configs", default=DEFAULT_CONFIGS,
                              help="Up to three beta,gamma,nlong,nshort configurations, separated by semicolons")
    trading_args.add_argument("--cost-rate", type=float, default=.0001,
                              help="Cost rate per traded notional at entry and exit (default: %(default)s)")
    trading_args.add_argument("--annual-risk-free-rate", type=float, default=.0381)
    trading_args.add_argument("--initial-cash", type=float, default=1_000_000.)

    execution_args = parser.add_argument_group("Run control")
    execution_args.add_argument("--step-timeout", type=float, default=None, help="Optional subprocess timeout in seconds; failures preserve logs")
    mode = execution_args.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and print the schedule without creating output or running agents")
    mode.add_argument("--resume", action="store_true", help="Resume an interrupted run from its committed checkpoints")
    args = parser.parse_args()
    try:
        args.input = args.input.resolve(strict=True)
        if args.candidate_file:
            args.candidate_file = args.candidate_file.resolve(strict=True)
        if args.calendar:
            args.calendar = args.calendar.resolve(strict=True)
        parse_configs(args.configs)
        if not np.isfinite(args.cost_rate) or args.cost_rate < 0:
            raise ValueError("cost-rate must be finite and non-negative")
        if not np.isfinite(args.initial_cash) or args.initial_cash <= 0:
            raise ValueError("initial-cash must be finite and positive")
        if not np.isfinite(args.annual_risk_free_rate) or args.annual_risk_free_rate <= -1:
            raise ValueError("annual-risk-free-rate must be finite and greater than -1")
        if args.max_pairs < 0 or args.max_temporal < 0:
            raise ValueError("Template candidate limits cannot be negative")
        if args.allow_feature_warmup_sessions < 0:
            raise ValueError("allow-feature-warmup-sessions cannot be negative")
        if args.allow_feature_warmup_sessions and not args.calendar:
            raise ValueError("allow-feature-warmup-sessions requires --calendar")
        if args.step_timeout is not None and (not np.isfinite(args.step_timeout) or args.step_timeout <= 0):
            raise ValueError("step-timeout must be finite and positive")
        pinned_inputs = None if args.dry_run else fingerprint_set(list(input_paths(args).values()))
        if args.require_explicit_label_dates:
            columns = pd.read_csv(args.input, nrows=0).columns
            missing = set(LABEL_END_COLUMNS.values()) - set(columns)
            if missing:
                raise ValueError(f"Strict label-date mode requires {sorted(missing)}; rebuild features from raw prices")
        calendar = pd.read_csv(args.input, usecols=["date"], parse_dates=["date"])["date"].drop_duplicates()
        plan = build_plan(calendar, args.initial_as_of, args.train_start, args.selection_start, args.selection_end,
                          args.holdout_start, args.holdout_end, args.maintenance_days, args.admission_sessions, args.min_train_sessions)
        if args.calendar:
            reference = read_calendar(args.calendar)
            audit = calendar_audit(calendar, reference, args.train_start, args.holdout_end)
            # Feature builders legitimately discard the leading indicator warm-up.
            # An explicit allowance only applies to a contiguous prefix, never
            # to missing sessions inside the experiment's observed feature span.
            first = pd.Timestamp(calendar.min())
            prefix = [date for date in audit["missing_sessions"] if pd.Timestamp(date) < first]
            if len(prefix) > args.allow_feature_warmup_sessions or len(audit["missing_sessions"]) != len(prefix) or audit["unexpected_sessions"]:
                raise ValueError(f"Feature calendar mismatch: missing={audit['missing_sessions'][:10]}, unexpected={audit['unexpected_sessions'][:10]}")
            audit["feature_warmup_sessions"] = len(prefix)
            audit["first_feature_session"] = str(first.date())
            audit["matches_reference_after_warmup"] = True
            plan["calendar_audit"] = audit
        else:
            plan["calendar_audit"] = {"mode": "observed_feature_sessions_only", "matches_reference": None}
        plan["require_explicit_label_dates"] = args.require_explicit_label_dates
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        summary = Pipeline(args, plan).run(pinned_inputs)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Pipeline failed: {exc}\n")
    print(json.dumps({key: summary[key] for key in ("status", "selected_config", "maintenance_rounds", "holdout")}, indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
