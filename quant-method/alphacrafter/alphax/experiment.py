"""Versioned records for AlphaX research runs."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4


RECORD_TYPE = "alphax.pipeline_run"
CSI300_RECORD = "alphax.csi300_factor_research"
CSI300_SCOPE = "retrospective_CSI300_factor_quality_not_executable_returns_or_certified_PIT"
CSI300_DATA_FILES = {"prices": "daily_pv.h5", "membership": "qlib_csi300.txt", "calendar": "qlib_day.txt"}
MEDIUM_RECORD = "alphax.medium_frequency_research"
MEDIUM_SCOPE = "retrospective_medium_frequency_research_not_live_execution"
EXPERIMENT_TYPES = {
    RECORD_TYPE: "pipeline", CSI300_RECORD: "csi300_factor_research", MEDIUM_RECORD: "medium_frequency_research",
}
SCHEMA_VERSION = 1
RUNTIME_PACKAGES = (
    "numpy", "pandas", "scipy", "python-dateutil", "pytz", "six", "typing-extensions", "tzdata",
)


def runtime_versions(extra_packages: tuple[str, ...] = ()) -> dict:
    dependencies = {}
    for name in (*RUNTIME_PACKAGES, *extra_packages):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "python": sys.version,
        "numpy": dependencies["numpy"],
        "pandas": dependencies["pandas"],
        "platform": sys.platform,
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def git_commit(checkout: Path) -> str | None:
    if not (checkout / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_manifest(
    parameters: dict, config: dict, inputs: dict, sources: dict, fingerprints: dict,
    plan: dict, checkout: Path, *, record_type: str = RECORD_TYPE,
    extra_packages: tuple[str, ...] = (), run_id: str | None = None,
) -> dict:
    def hashes(files: dict) -> dict:
        return {name: record["sha256"] for name, record in files.items()}

    runtime = runtime_versions(extra_packages)
    conditions = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": plan["runner_version"],
        "config": config,
        "inputs": hashes(inputs),
        "code": hashes(sources),
        "runtime": runtime,
    }
    return {
        "record_type": record_type,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id if run_id is not None else str(uuid4()),
        "experiment_fingerprint": _fingerprint(conditions),
        "parameters": parameters,
        "config": config,
        "fingerprints": fingerprints,
        "inputs": inputs,
        "code": {"git_commit": git_commit(checkout), "files": sources},
        "runtime": runtime,
        "selection": plan["selection"],
        "holdout": plan["holdout"],
        "candidate_provenance": (
            "External fixed hypotheses are user-attested, not certified point-in-time"
            if "candidate_file" in inputs else "Deterministic template generator"
        ),
        "records": {"status": "status.json", "summary": "summary.json", "artifacts": "artifacts.json"},
    }


def artifact_inventory(directory: Path) -> dict:
    """List files available at finalization, relative to the run directory."""
    files = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if relative not in {"artifacts.json", "status.json"} and not path.is_symlink() and path.is_file():
            files.append(relative)
    return {"schema_version": SCHEMA_VERSION, "files": sorted(files)}


def _read_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _record_object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _data_version(manifest: dict, record_type: str, supported: bool) -> str | None:
    inputs = _record_object(manifest.get("inputs"))
    if record_type in (CSI300_RECORD, MEDIUM_RECORD):
        if not supported:
            if manifest.get("record_type"):
                return None
            key = "files" if record_type == CSI300_RECORD else "sources"
            files = {Path(path).name: entry for path, entry in _record_object(manifest.get(key)).items()}
            inputs = {role: files.get(name) for role, name in CSI300_DATA_FILES.items()}
        hashes = {role: _record_object(inputs.get(role)).get("sha256") for role in CSI300_DATA_FILES}
        return _fingerprint(hashes) if all(hashes.values()) else None
    if supported:
        return _record_object(inputs.get("panel")).get("sha256")
    if not manifest.get("record_type"):
        source = _record_object(manifest.get("parameters")).get("input")
        if isinstance(source, str):
            return _record_object(_record_object(manifest.get("fingerprints")).get(source)).get("sha256")
    return None


def _interval(value: object) -> dict | None:
    return dict(zip(("start", "end"), value)) if isinstance(value, list) and len(value) == 2 else None


def _medium_details(manifest: dict, summary: dict, selection: dict, completed: bool) -> dict:
    config = _record_object(manifest.get("config"))
    evaluated = config.get("evaluate_strategies")
    if evaluated is None and manifest.get("visible_end") in ("2021-12-31", "2023-12-31"):
        evaluated = manifest["visible_end"] == "2023-12-31"
    eligible = _record_object(summary.get("research_eligible"))
    combinations = _record_object(summary.get("eligible_combinations"))
    trials = summary.get("strategies")
    by_horizon = {}
    for horizon in (1, 5, 20):
        names = combinations.get(str(horizon))
        by_horizon[str(horizon)] = {
            "eligible_factors": eligible.get(str(horizon)),
            "eligible_combinations": len(names) if isinstance(names, list) else None,
            "trials": [
                {"strategy_id": trial.get("strategy_id"), "status": trial.get("status")}
                for trial in trials if isinstance(trial, dict) and trial.get("horizon") == horizon
            ] if isinstance(trials, list) else None,
        }
    miner = _record_object(manifest.get("miner_source"))
    return {
        "run_mode": "strategy_comparison" if evaluated else "factor_research" if evaluated is False else None,
        "weekly_xgboost": {
            "comparison": config.get("compare_weekly_xgboost", "weekly_xgboost_comparison" in summary if completed else None),
            "additions": config.get("compare_weekly_xgboost_additions", "weekly_xgboost_additions" in summary if completed else None),
        },
        "miner_source": {"generator": miner.get("generator", manifest.get("miner_generator")), "as_of": miner.get("as_of")},
        "strategy_iteration": _interval(manifest.get("strategy_iteration")),
        "strategy_iteration_previously_examined": manifest.get("strategy_iteration_previously_examined"),
        "final_reserved": _interval(manifest.get("final_reserved")),
        "strategy_status": selection.get("status"),
        "metrics": {
            "by_horizon": by_horizon, "selected_horizon": selection.get("selected_horizon"),
            "selected_strategy_id": selection.get("selected_strategy_id"),
            "holdout_evaluated": summary.get("holdout_evaluated"),
        },
    }


def _catalog_run(directory: Path, manifest: dict, plan: dict, status: dict, record_type: str) -> dict:
    versioned = manifest.get("record_type") == record_type
    supported = versioned and manifest.get("schema_version") == SCHEMA_VERSION
    state = status.get("status")
    is_research = record_type != RECORD_TYPE
    completed = state == ("completed_research" if is_research else "completed")
    summary = _read_record(directory / "summary.json") if completed else {}
    details = {}
    if record_type == CSI300_RECORD:
        accepted = summary.get("strict_accepted")
        metrics = {
            "candidate_count": summary.get("candidate_count"),
            "strict_accepted_count": len(accepted) if isinstance(accepted, list) else None,
            "holdout_evaluated": summary.get("holdout_evaluated"),
        }
        holdout = _interval(manifest.get("holdout"))
    elif record_type == MEDIUM_RECORD:
        selection = _read_record(directory / "strategy_selection.json") if completed else {}
        details = _medium_details(manifest, summary, selection, completed)
        holdout, metrics = None, details.pop("metrics")
    else:
        holdout = (manifest.get("holdout") if supported else None) or plan.get("holdout")
        results = _record_object(summary.get("holdout"))
        metrics = {key: results.get(key) for key in ("cumulative_return", "sharpe", "max_drawdown")}
    own_status = status.get("record_type") == record_type
    inventory = _read_record(directory / "artifacts.json") if supported or (not versioned and own_status) else {}
    return {
        "directory": str(directory),
        "experiment_type": EXPERIMENT_TYPES[record_type],
        "schema_version": manifest.get("schema_version") if versioned else None,
        "run_id": manifest.get("run_id") if supported else status.get("run_id") if own_status else None,
        "experiment_fingerprint": manifest.get("experiment_fingerprint") if supported else None,
        "status": "completed" if completed else "failed" if state == "failed" else "unknown",
        "failed_stage": status.get("stage") if state == "failed" else None,
        "error": status.get("error") if state == "failed" else None,
        "data_version": _data_version(manifest, record_type, supported),
        "holdout": holdout,
        "strategy_status": summary.get("status"),
        "metrics": metrics,
        "artifacts": inventory.get("files") if inventory.get("schema_version") == SCHEMA_VERSION else None,
        **details,
    }


def list_runs(root: Path) -> list[dict]:
    """Find AlphaX experiment runs, including supported legacy records."""
    if not root.is_dir():
        raise ValueError(f"Experiment directory does not exist: {root}")
    runs = []
    for name, directories, files in os.walk(root.resolve()):
        directories.sort()
        if not {"manifest.json", "plan.json", "status.json"}.intersection(files):
            continue
        directory = Path(name)
        manifest = _read_record(directory / "manifest.json")
        plan = _read_record(directory / "plan.json")
        status = _read_record(directory / "status.json")
        record_type = manifest.get("record_type") or status.get("record_type")
        if not record_type and plan.get("runner_version") == "pit-pipeline-v1":
            record_type = RECORD_TYPE
        if not record_type:
            record_type = {CSI300_SCOPE: CSI300_RECORD, MEDIUM_SCOPE: MEDIUM_RECORD}.get(str(manifest.get("scope")))
        if isinstance(record_type, str) and record_type in EXPERIMENT_TYPES:
            runs.append(_catalog_run(directory, manifest, plan, status, record_type))
    return runs
