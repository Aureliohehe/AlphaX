"""Versioned experiment records for the chronological AlphaCrafter pipeline."""

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
SCHEMA_VERSION = 1
RUNTIME_PACKAGES = (
    "numpy", "pandas", "scipy", "python-dateutil", "pytz", "six", "typing-extensions", "tzdata",
)


def runtime_versions() -> dict:
    dependencies = {}
    for name in RUNTIME_PACKAGES:
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


def build_manifest(
    parameters: dict, config: dict, inputs: dict, sources: dict, fingerprints: dict,
    plan: dict, checkout: Path,
) -> dict:
    def hashes(files: dict) -> dict:
        return {name: record["sha256"] for name, record in files.items()}

    runtime = runtime_versions()
    conditions = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": plan["runner_version"],
        "config": config,
        "inputs": hashes(inputs),
        "code": hashes(sources),
        "runtime": runtime,
    }
    encoded = json.dumps(conditions, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "record_type": RECORD_TYPE,
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid4()),
        "experiment_fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
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


def _catalog_run(directory: Path, manifest: dict, plan: dict) -> dict:
    versioned = manifest.get("record_type") == RECORD_TYPE
    supported = versioned and manifest.get("schema_version") == SCHEMA_VERSION
    status = _read_record(directory / "status.json")
    state = status.get("status")
    summary = _read_record(directory / "summary.json") if state == "completed" else {}
    metrics = _record_object(summary.get("holdout"))
    inventory = _read_record(directory / "artifacts.json") if supported else {}
    panel = {}
    if supported:
        panel = _record_object(_record_object(manifest.get("inputs")).get("panel"))
    elif not versioned:
        source = _record_object(manifest.get("parameters")).get("input")
        if isinstance(source, str):
            panel = _record_object(_record_object(manifest.get("fingerprints")).get(source))
    return {
        "directory": str(directory),
        "schema_version": manifest.get("schema_version") if versioned else None,
        "run_id": manifest.get("run_id") if supported else None,
        "experiment_fingerprint": manifest.get("experiment_fingerprint") if supported else None,
        "status": state if state in ("completed", "failed") else "unknown",
        "failed_stage": status.get("stage") if state == "failed" else None,
        "error": status.get("error") if state == "failed" else None,
        "data_version": panel.get("sha256"),
        "holdout": (manifest.get("holdout") if supported else None) or plan.get("holdout"),
        "strategy_status": summary.get("status"),
        "metrics": {key: metrics.get(key) for key in ("cumulative_return", "sharpe", "max_drawdown")},
        "artifacts": inventory.get("files") if inventory.get("schema_version") == SCHEMA_VERSION else None,
    }


def list_runs(root: Path) -> list[dict]:
    """Find pipeline runs, including legacy records, without modifying them."""
    if not root.is_dir():
        raise ValueError(f"Experiment directory does not exist: {root}")
    runs = []
    for name, directories, files in os.walk(root.resolve()):
        directories.sort()
        if "manifest.json" not in files and "plan.json" not in files:
            continue
        directory = Path(name)
        manifest = _read_record(directory / "manifest.json")
        plan = _read_record(directory / "plan.json")
        if manifest.get("record_type") == RECORD_TYPE or plan.get("runner_version") == "pit-pipeline-v1":
            runs.append(_catalog_run(directory, manifest, plan))
    return runs
