"""Shared feature/label metadata and session-calendar contract.

Label dates are evaluation metadata, NEVER model features. Explicit dates
survive feature-row filtering. Legacy inference is kept only as an auditable
compatibility mode; a supplied calendar is a reference, not proof of provenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

LABEL_END_COLUMNS = {1: "label_end_1d", 5: "label_end_5d"}
TARGET_COLUMNS = {1: ("target_1d", "target_open_close"), 5: ("target_5d", "target_open_close_5d")}
META_COLUMNS = {"date", "ticker", "split", *LABEL_END_COLUMNS.values(),
                *(column for columns in TARGET_COLUMNS.values() for column in columns)}


def is_feature_column(name: str) -> bool:
    return name not in META_COLUMNS and not name.startswith(("label_end_", "target_"))


def parse_session_series(values: pd.Series, name: str, allow_missing: bool = False) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
        if parsed.dt.tz is not None:
            raise ValueError("timezones are not session dates")
        if (parsed.dropna() != parsed.dropna().dt.normalize()).any():
            raise ValueError("intraday timestamps are not session dates")
        if not allow_missing and parsed.isna().any():
            raise ValueError("missing session dates")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    return parsed


def validate_calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = parse_session_series(pd.Series(calendar), "calendar")
    if dates.empty or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Calendar must be non-empty, unique and strictly increasing")
    return pd.DatetimeIndex(dates)


def read_calendar(path: Path) -> pd.DatetimeIndex:
    return validate_calendar(pd.DatetimeIndex(parse_session_series(pd.read_csv(path, usecols=["date"])["date"], "calendar")))


def calendar_audit(
    observed: pd.Series, reference: pd.DatetimeIndex, start: object, end: object,
) -> Dict[str, object]:
    reference = validate_calendar(reference)
    bounds = parse_session_series(pd.Series([start, end]), "audit bounds")
    start, end = bounds.iloc[0], bounds.iloc[1]
    if start > end or reference[0] > start or reference[-1] < end:
        raise ValueError("Reference calendar must cover the requested audit bounds (include padding sessions)")
    observed = pd.DatetimeIndex(parse_session_series(pd.Series(observed), "observed calendar").unique())
    expected = reference[(reference >= start) & (reference <= end)]
    actual = observed[(observed >= start) & (observed <= end)]
    missing, unexpected = expected.difference(actual), actual.difference(expected)
    return {
        "start": str(start.date()), "end": str(end.date()),
        "expected_sessions": len(expected), "observed_sessions": len(actual),
        "missing_sessions": [str(date.date()) for date in missing],
        "unexpected_sessions": [str(date.date()) for date in unexpected],
        "matches_reference": len(missing) == 0 and len(unexpected) == 0,
        "provenance": "supplied reference; calendar authority and constituent membership are not certified",
    }


def label_end_source(frame: pd.DataFrame, horizon: int) -> str:
    return "explicit_raw_label_dates" if LABEL_END_COLUMNS.get(horizon) in frame else "legacy_feature_row_inference"


def resolve_label_ends(
    frame: pd.DataFrame, horizon: int, require_explicit: bool = False,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    if horizon not in LABEL_END_COLUMNS:
        raise ValueError("Supported label horizons are 1 and 5 sessions")
    name = LABEL_END_COLUMNS[horizon]
    signal_dates = parse_session_series(frame["date"], "signal dates")
    if name not in frame:
        if require_explicit:
            raise ValueError(f"Missing {name}; rebuild features from raw prices (strict label-date mode)")
        ordered = frame.sort_values(["ticker", "date"])
        ends = ordered.groupby("ticker", sort=False)["date"].shift(-horizon).reindex(frame.index)
    else:
        ends = parse_session_series(frame[name], name, allow_missing=True)
    if (ends.notna() & (ends <= signal_dates)).any():
        raise ValueError(f"{name} must be strictly after its signal date")
    for target in TARGET_COLUMNS[horizon]:
        if name in frame and target in frame and (frame[target].notna() & ends.isna()).any():
            raise ValueError(f"Non-missing {target} has no explicit {name}")
    if horizon == 5 and LABEL_END_COLUMNS[1] in frame:
        first = parse_session_series(frame[LABEL_END_COLUMNS[1]], LABEL_END_COLUMNS[1], allow_missing=True)
        if (ends.notna() & (first.isna() | (ends <= first))).any():
            raise ValueError("Five-session label end must follow the one-session label end")
    if calendar is not None:
        calendar = validate_calendar(calendar)
        if not signal_dates.isin(calendar).all():
            raise ValueError("Signal dates outside reference calendar")
        expected = signal_dates.map(pd.Series(calendar, index=calendar).shift(-horizon))
        if (ends.notna() & (expected.isna() | (ends != expected))).any():
            raise ValueError(f"{name} does not match reference market sessions; rebuild labels with --calendar")
    return ends


def forward_labels(
    dates: pd.Series, close: pd.Series, open_: pd.Series,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Construct values AND dates before any feature warm-up/missingness filter.

    Without a calendar, preserve the existing per-instrument raw-row horizon.
    With a calendar, h means market sessions, not the next h available quotes;
    missing prices yield missing targets, never a shift to a later quote.
    """
    dates = parse_session_series(dates, "raw dates")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Raw label dates must be unique and increasing within an instrument")
    ends = {}
    if calendar is None:
        ends = {h: dates.shift(-h) for h in LABEL_END_COLUMNS}
    else:
        calendar = validate_calendar(calendar)
        if not dates.isin(calendar).all():
            raise ValueError("Raw quote dates outside supplied calendar")
        for horizon in LABEL_END_COLUMNS:
            successors = pd.Series(calendar, index=calendar).shift(-horizon)
            ends[horizon] = dates.map(successors)
    close_by_date = pd.Series(close.to_numpy(), index=dates)
    open_by_date = pd.Series(open_.to_numpy(), index=dates)
    future_close = {h: ends[h].map(close_by_date) for h in ends}
    next_open = ends[1].map(open_by_date).replace(0, np.nan)
    result = pd.DataFrame({
        "target_1d": future_close[1] / close.replace(0, np.nan) - 1,
        "target_5d": future_close[5] / close.replace(0, np.nan) - 1,
        "target_open_close": future_close[1] / next_open - 1,
        "target_open_close_5d": future_close[5] / next_open - 1,
        "label_end_1d": ends[1], "label_end_5d": ends[5],
    }, index=dates.index)
    result[list(TARGET_COLUMNS[1]) + list(TARGET_COLUMNS[5])] = result[list(TARGET_COLUMNS[1]) + list(TARGET_COLUMNS[5])].replace([np.inf, -np.inf], np.nan)
    return result
