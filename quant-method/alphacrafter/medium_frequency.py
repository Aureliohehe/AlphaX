"""Executable-horizon factors and maturity-aware screening for 5/20-session research."""
import numpy as np
import pandas as pd

from cn_portfolio import CNConfig


HORIZONS = (5, 20)
FACTOR_HYPOTHESES = {
    "mf_momentum_60_skip5": "Intermediate momentum excluding the most recent week",
    "mf_momentum_120_skip20": "Longer momentum excluding recent monthly reversal",
    "mf_trend_20_60": "Medium moving-average trend strength",
    "mf_trend_60_120": "Longer moving-average trend strength",
    "mf_low_volatility_60": "Persistent low realized volatility",
    "mf_low_downside_risk_60": "Persistent low downside return variation",
    "mf_risk_adjusted_momentum_60": "Intermediate momentum relative to realized risk",
    "mf_drawdown_60": "Distance from a trailing quarterly high",
    "mf_volume_stability_60": "Stable trading participation over a quarter",
    "mf_volume_confirmed_trend": "Medium trend confirmed by relative volume",
    "mf_trend_consistency_60": "Persistence in the direction of daily price changes",
    "mf_reversal_20": "Monthly price displacement may mean revert",
}


def definitions():
    return [{"candidate_id": name, "features": [name], "expression": {"op": "feature", "feature": name},
             "hypothesis": hypothesis} for name, hypothesis in FACTOR_HYPOTHESES.items()]


def strategy_presets():
    return {5: CNConfig(30, rebalance_interval=5, min_holding_sessions=5, sell_rank_buffer=30,
                        max_daily_turnover=.10, max_daily_cost=.0003, rebalance_existing=False),
            20: CNConfig(30, rebalance_interval=20, min_holding_sessions=20, sell_rank_buffer=30,
                         max_daily_turnover=.20, max_daily_cost=.0005, rebalance_existing=False)}


def execution_labels(dates, open_, calendar, horizons=HORIZONS):
    """T-close signal; enter T+1 open, exit T+h+1 open; no delayed-quote shift."""
    dates = pd.Series(pd.to_datetime(dates), index=dates.index)
    calendar = pd.DatetimeIndex(calendar)
    if dates.duplicated().any() or not dates.is_monotonic_increasing or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Labels require unique increasing raw dates and calendar")
    if not dates.isin(calendar).all():
        raise ValueError("Raw dates outside calendar")
    opens = pd.Series(open_.to_numpy(), index=dates).where(lambda s: s > 0)
    successors = pd.Series(calendar, index=calendar)
    entry = dates.map(successors.shift(-1))
    result = pd.DataFrame(index=dates.index)
    for h in horizons:
        if isinstance(h, bool) or not isinstance(h, int) or h < 1:
            raise ValueError("Invalid execution horizon")
        end = dates.map(successors.shift(-(h + 1)))
        result[f"label_start_{h}d"] = entry
        result[f"label_end_{h}d"] = end
        result[f"target_open_open_{h}d"] = end.map(opens) / entry.map(opens) - 1
    return result.replace([np.inf, -np.inf], np.nan)


def factor_panel(raw, calendar, end=None):
    """Use complete stock history before membership filtering; unknown bars remain unknown."""
    if raw.ticker.nunique() != 1 or raw.date.duplicated().any():
        raise ValueError("Expected one stock and unique sessions")
    raw = raw.sort_values("date")
    dates = calendar[(calendar >= raw.date.min()) & (calendar <= (pd.Timestamp(end) if end is not None else raw.date.max()))]
    ticker = raw.ticker.iloc[0]
    frame = raw.set_index("date").reindex(dates).rename_axis("date").reset_index()
    frame["ticker"] = ticker
    for c in ("open", "close", "high", "low", "volume"):
        frame[c] = frame[c].where(frame[c] > 0)
    close, volume = frame.close, frame.volume
    ret = np.log(close).diff()
    vol = ret.rolling(60).std()
    out = frame[["date", "ticker"]].copy()
    out["mf_momentum_60_skip5"] = close.shift(5) / close.shift(60) - 1
    out["mf_momentum_120_skip20"] = close.shift(20) / close.shift(120) - 1
    out["mf_trend_20_60"] = close.rolling(20).mean() / close.rolling(60).mean() - 1
    out["mf_trend_60_120"] = close.rolling(60).mean() / close.rolling(120).mean() - 1
    out["mf_low_volatility_60"] = -vol
    out["mf_low_downside_risk_60"] = -np.sqrt(ret.clip(upper=0).pow(2).rolling(60).mean())
    out["mf_risk_adjusted_momentum_60"] = (close / close.shift(60) - 1) / (vol * np.sqrt(60)).replace(0, np.nan)
    out["mf_drawdown_60"] = close / frame.high.rolling(60).max() - 1
    out["mf_volume_stability_60"] = -np.log(volume).diff().rolling(60).std()
    out["mf_volume_confirmed_trend"] = out.mf_trend_20_60 * volume.rolling(20).mean() / volume.rolling(60).mean()
    out["mf_trend_consistency_60"] = np.sign(ret).rolling(60).mean()
    out["mf_reversal_20"] = -(close / close.shift(20) - 1)
    current_valid = frame[["open", "close", "high", "low", "volume"]].notna().all(axis=1)
    out.loc[~current_valid, list(FACTOR_HYPOTHESES)] = np.nan
    return pd.concat([out, execution_labels(frame.date, frame.open, calendar)], axis=1).replace([np.inf, -np.inf], np.nan)


def anchor_mask(dates, calendar, first_execution, horizon):
    """Fixed rebalance phase, also used for nonoverlapping research labels."""
    positions = calendar.get_indexer(pd.DatetimeIndex(dates))
    origin = calendar.get_loc(pd.Timestamp(first_execution))
    return pd.Series((positions >= 0) & ((positions + 1 - origin) % horizon == 0), index=dates.index)


def daily_rank_ic(frame, score, target):
    values = pd.DataFrame({"date": frame.date, "score": score, "target": target}).dropna()
    ranks = values.groupby("date")[["score", "target"]].rank()
    centered = ranks - ranks.groupby(values.date).transform("mean")
    numerator = (centered.score * centered.target).groupby(values.date).sum()
    denominator = np.sqrt(centered.score.pow(2).groupby(values.date).sum() * centered.target.pow(2).groupby(values.date).sum())
    counts = values.groupby("date").size()
    return (numerator / denominator.replace(0, np.nan)).where(counts >= 3).dropna()


def screen_signals(frame, pool, horizon, calendar, first_execution, start, end,
                   lookback_periods=12, min_periods=6, min_rank_ic=0.):
    """Screener learns only from completed matching-horizon, nonoverlapping events."""
    if lookback_periods < min_periods or min_periods < 1 or horizon not in HORIZONS:
        raise ValueError("Invalid medium-frequency screening settings")
    if pool.get("label_mode") != "execution_open_open":
        raise ValueError("Medium-frequency screening requires open-to-open labels")
    if pool.get("artifact_type") != "research_candidate_pool_v1" or pool.get("market") != "cn":
        raise ValueError("Expected a CN research candidate pool")
    cutoffs = [pd.Timestamp(pool["research_end"]), pd.Timestamp(pool["as_of"])]
    if any(pd.isna(d) for d in cutoffs) or pd.Timestamp(start) <= max(cutoffs):
        raise ValueError("Strategy screening must follow the completed research cutoff")
    entries = pool["candidates_by_horizon"].get(str(horizon), [])
    if not entries:
        return pd.DataFrame(columns=["date", "ticker", "prediction"]), pd.DataFrame(columns=["date", "selected_factors"])
    scores, events = {}, {}
    anchors = anchor_mask(frame.date, calendar, first_execution, horizon)
    label = frame[f"target_open_open_{horizon}d"]
    ends = frame.groupby("date")[f"label_end_{horizon}d"].first()
    for entry in entries:
        name, direction = entry["candidate_id"], entry["direction"]
        score = direction * frame[entry["definition"]["features"][0]]
        scores[name] = score.groupby(frame.date).rank(pct=True)
        ic = daily_rank_ic(frame, score.where(anchors), label)
        events[name] = pd.DataFrame({"source": ic.index, "end": ends.reindex(ic.index).to_numpy(), "ic": ic.to_numpy()}).dropna()
    first = calendar[calendar >= pd.Timestamp(start)][0]
    previous = dict(zip(calendar[1:], calendar[:-1]))
    source_start = previous[first]
    output, selections = [], []
    for date, group in frame.loc[anchors & frame.date.between(source_start, end)].groupby("date", sort=True):
        selected, last_end = [], None
        for name, table in events.items():
            available = table.loc[table.end <= date].sort_values("source").tail(lookback_periods)
            if len(available) >= min_periods and available.ic.mean() > min_rank_ic:
                selected.append(name)
                latest = available.end.max()
                last_end = latest if last_end is None else max(last_end, latest)
        selections.append({"date": date, "horizon": horizon, "selected_factors": ",".join(selected),
                           "count": len(selected), "last_label_end_used": last_end})
        if selected:
            combined = pd.concat([scores[name].loc[group.index] for name in selected], axis=1)
            prediction = combined.sum(axis=1, min_count=len(selected)) / len(selected)
            output.append(pd.DataFrame({"date": group.date, "ticker": group.ticker, "prediction": prediction}))
    result = pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=["date", "ticker", "prediction"])
    return result, pd.DataFrame(selections)


def load_member_bars(source, membership, start, end):
    import h5py
    intervals = pd.read_csv(membership, sep="\t", header=None, names=["ticker", "start", "end"], parse_dates=["start", "end"])
    relevant = intervals[(intervals.end >= pd.Timestamp(start)) & (intervals.start <= pd.Timestamp(end))]
    pieces = []
    with h5py.File(source) as hdf:
        g = hdf["data"]
        fields = [v.decode().lstrip("$") for v in g["axis0"][:]]
        if fields != ["open", "close", "high", "low", "volume", "factor"]:
            raise ValueError("Unexpected HDF field order")
        dates = pd.DatetimeIndex(pd.to_datetime(g["axis1_level0"][:]))
        names = np.array([v.decode() for v in g["axis1_level1"][:]])
        dl, il = g["axis1_label0"][:], g["axis1_label1"][:]
        codes = np.isin(names, relevant.ticker.unique())
        for i in range(0, len(dl), 250000):
            d, code = dl[i:i+250000], il[i:i+250000]
            mask = codes[code] & (dates[d] <= pd.Timestamp(end))
            if mask.any():
                values = g["block0_values"][i:i+250000][mask]
                part = pd.DataFrame(values[:, :5], columns=fields[:5])
                part["date"], part["ticker"] = dates[d[mask]], names[code[mask]]
                pieces.append(part)
    return pd.concat(pieces, ignore_index=True), relevant
