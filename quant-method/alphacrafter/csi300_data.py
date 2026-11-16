"""Adapt supplied adjusted Qlib OHLCV without filling suspension prices."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1] / "machine-learn"
sys.path.insert(0, str(ML))
from build_features import add_features


def member_mask(dates, intervals):
    result = pd.Series(False, index=dates.index)
    for row in intervals.itertuples():
        result |= dates.between(row.start, row.end)
    return result


def ticker_features(raw, calendar, intervals, columns, start, end):
    """Warm on all prior sessions; only then select historical member rows."""
    ticker = raw["ticker"].iloc[0]
    raw = raw.sort_values("date")
    if raw.date.duplicated().any():
        raise ValueError("Duplicate raw instrument dates")
    sessions = calendar[(calendar >= raw.date.min()) & (calendar <= pd.Timestamp(end))]
    raw = raw.set_index("date").reindex(sessions).rename_axis("date").reset_index()
    raw["ticker"] = ticker
    # Input OHLC is already adjusted/normalized. Keep adjustment exactly one.
    # Nonpositive or missing source prices are unknown, never forward-filled.
    for column in ("open", "high", "low", "close", "volume"):
        raw[column] = raw[column].where(raw[column] > 0)
    raw["adj_close"] = raw["close"]
    result = add_features(raw, calendar=calendar)
    for window in (1, 5, 10, 20, 60):
        result[f"return_{window}d"] = raw.close.pct_change(window, fill_method=None)
    result["volume_change_1d"] = raw.volume.pct_change(fill_method=None)
    # Count-based Alpha158 features otherwise treat missing deltas as down/up=0.
    valid_delta = raw.close.diff().notna().rolling(20).sum().eq(20)
    for column in ("alpha158_cntd20", "alpha158_sumd20"):
        if column in result:
            result.loc[~valid_delta, column] = np.nan
    current_valid = raw[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    feature_columns = [c for c in columns if c not in ("date", "ticker") and not c.startswith(("target_", "label_end_"))]
    result.loc[~current_valid, feature_columns] = np.nan
    keep = member_mask(result.date, intervals) & result.date.between(start, end)
    # Keep unknown features/targets in the universe so coverage cannot improve
    # by deleting suspended names or names without future prices.
    return result.loc[keep, columns].replace([np.inf, -np.inf], np.nan)


def build_panel(source, membership, calendar, output, columns, start, end):
    import h5py

    intervals = pd.read_csv(membership, sep="\t", header=None,
                            names=["instrument", "start", "end"], parse_dates=["start", "end"])
    relevant = intervals[(intervals.end >= pd.Timestamp(start)) & (intervals.start <= pd.Timestamp(end))]
    pieces = []
    with h5py.File(source) as hdf:
        group = hdf["data"]
        fields = [value.decode().lstrip("$") for value in group["axis0"][:]]
        if fields != ["open", "close", "high", "low", "volume", "factor"]:
            raise ValueError("Unexpected Qlib HDF field order")
        dates = pd.DatetimeIndex(pd.to_datetime(group["axis1_level0"][:]))
        names = np.array([value.decode() for value in group["axis1_level1"][:]])
        dlabels, ilabels = group["axis1_label0"][:], group["axis1_label1"][:]
        eligible_codes = np.isin(names, relevant.instrument.unique())
        for offset in range(0, len(dlabels), 250000):
            dl, il = dlabels[offset:offset+250000], ilabels[offset:offset+250000]
            mask = eligible_codes[il] & (dates[dl] <= pd.Timestamp(end))
            if not mask.any():
                continue
            values = group["block0_values"][offset:offset+250000][mask]
            part = pd.DataFrame(values[:, :5], columns=fields[:5])
            part["date"], part["ticker"] = dates[dl[mask]], names[il[mask]]
            pieces.append(part)
    raw = pd.concat(pieces, ignore_index=True)
    del pieces
    grouped = raw.groupby("ticker", sort=True)
    wrote = False
    row_count = 0
    # Each append is a separate valid gzip member, readable by pandas/gzip.
    for index, (ticker, group) in enumerate(grouped, 1):
        result = ticker_features(group, calendar, relevant[relevant.instrument == ticker], columns, start, end)
        result.to_csv(output, index=False, mode="a" if wrote else "w", header=not wrote,
                      compression={"method": "gzip", "compresslevel": 1})
        wrote = True
        row_count += len(result)
        if index % 100 == 0:
            print(f"Features: {index}/{len(grouped)} instruments; {row_count} member rows", flush=True)
    if not wrote:
        raise ValueError("No member features produced")
    return {"member_rows": row_count, "instruments_with_source_rows": len(grouped),
            "universe": "supplied membership intervals; missing feature/target rows retained",
            "adjustment": "source-adjusted OHLC used as-is; factor excluded"}
