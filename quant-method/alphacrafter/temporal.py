"""Shared point-in-time boundaries for Miner admission and Screener replay.

Dates denote information available at that session's close. Prefer explicit
label dates saved from raw quotes before feature filtering. Legacy inference
is only a documented compatibility path and can over-purge filtered panels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# The repository uses standalone scripts in sibling, hyphenated directories.
# Keep the data contract shared with feature generation and ML consumers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "machine-learn"))
from data_contract import (  # noqa: E402
    LABEL_END_COLUMNS, META_COLUMNS, calendar_audit, is_feature_column,
    label_end_source, read_calendar, resolve_label_ends,
)

TEMPORAL_POLICY_VERSION = "label-end-v1"


def session_date(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"Expected a timezone-naive session date, got {value!r}")
    return stamp


def visible_panel(frame: pd.DataFrame, as_of: object) -> pd.DataFrame:
    cutoff = session_date(as_of)
    if frame[["date", "ticker"]].isna().any().any() or frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Feature panel requires unique, non-null date/ticker keys")
    visible = frame.loc[frame["date"] <= cutoff].sort_values(["date", "ticker"]).reset_index(drop=True)
    if visible.empty:
        raise ValueError(f"No feature rows visible at {cutoff.date()}")
    return visible


def label_end_dates(
    frame: pd.DataFrame, horizon: int, require_explicit: bool = False,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    return resolve_label_ends(frame, horizon, require_explicit, calendar)


def validate_windows(windows: Dict[str, Tuple[str, str]]) -> None:
    previous_end = None
    for name in ("train", "validation", "backtest"):
        start, end = map(session_date, windows[name])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Require ordered, non-overlapping train, validation and backtest windows")
        previous_end = end


def purged_mask(
    dates: pd.Series, label_ends: pd.Series, start: object, end: object, as_of: object,
) -> pd.Series:
    cutoff = min(session_date(end), session_date(as_of))
    return dates.between(session_date(start), cutoff) & label_ends.notna() & (label_ends <= cutoff)


def snapshot_date(payload: Dict[str, object], require_policy: bool = True) -> pd.Timestamp:
    if require_policy and payload.get("temporal_policy_version") != TEMPORAL_POLICY_VERSION:
        raise ValueError("Unverified legacy library: rebuild with the time-isolated Miner, or explicitly request retrospective diagnostics")
    if not payload.get("as_of"):
        raise ValueError("Factor snapshot requires an as_of date")
    cutoff = session_date(payload["as_of"])
    factors = payload.get("factors", payload.get("accepted_factors", []))
    if not isinstance(factors, list):
        raise ValueError("Snapshot factors must be a list")
    ids = set()
    for factor in factors:
        if not isinstance(factor, dict) or not factor.get("candidate_id"):
            raise ValueError("Every snapshot factor requires a candidate_id")
        if factor["candidate_id"] in ids:
            raise ValueError("Duplicate factor ID in snapshot")
        ids.add(factor["candidate_id"])
        created = session_date(factor.get("created_at"))
        validated = session_date(factor.get("last_validated_at"))
        if not created <= validated <= cutoff:
            raise ValueError("Factor lifecycle timestamps exceed or contradict snapshot as_of")
        for entry in factor.get("validation_history", []):
            if session_date(entry.get("validated_at")) > cutoff:
                raise ValueError("Future factor validation history in snapshot")
    return cutoff


def validate_previous_registry(payload: Dict[str, object], as_of: object) -> None:
    if not payload:
        return
    if snapshot_date(payload) > session_date(as_of):
        raise ValueError("Cannot reuse a future registry for an earlier as_of run; use a fresh output directory")
    for entry in payload.get("run_history", []):
        if session_date(entry.get("as_of")) > session_date(as_of):
            raise ValueError("Future run history in registry")


def write_snapshot(directory: Path, payload: Dict[str, object]) -> Path:
    """Write-once archive: never replace a different snapshot for the same date."""
    cutoff = snapshot_date(payload)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cutoff.date()}.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(text)
    except FileExistsError:
        if path.read_text() != text:
            raise ValueError(f"Immutable snapshot already exists with different content: {path}")
    return path


def load_timeline(directory: Path) -> List[Tuple[pd.Timestamp, Dict[str, object]]]:
    timeline = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Snapshot must be an object: {path}")
        timeline.append((snapshot_date(payload), payload))
    if not timeline:
        raise ValueError(f"No snapshots found in {directory}")
    timeline.sort(key=lambda item: item[0])
    if len({date for date, _ in timeline}) != len(timeline):
        raise ValueError("Multiple snapshots have the same as_of date")
    return timeline


def active_factors_at(
    timeline: List[Tuple[pd.Timestamp, Dict[str, object]]], date: pd.Timestamp,
) -> List[Dict[str, object]]:
    latest = None
    for cutoff, payload in timeline:
        if cutoff > date:
            break
        latest = payload
    if latest is None:
        return []
    # An empty snapshot means the library was pruned, not 'reuse the old one'.
    return [factor for factor in latest.get("factors", latest.get("accepted_factors", []))
            if factor.get("status", "active") == "active"]
