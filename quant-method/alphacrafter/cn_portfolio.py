"""Persistent long-only adjusted-unit research simulator, not broker execution."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CNConfig:
    topk: int
    exposure: float = .8
    commission: float = .0003
    slippage: float = .0005
    opening_gap_guard: float = .048

    def __post_init__(self):
        if isinstance(self.topk, bool) or not isinstance(self.topk, int) or self.topk < 1:
            raise ValueError("topk must be a positive integer")
        if not all(np.isfinite(v) for v in (self.exposure, self.commission, self.slippage, self.opening_gap_guard)):
            raise ValueError("Nonfinite execution configuration")
        if not 0 < self.exposure <= 1 or min(self.commission, self.slippage) < 0 or not 0 < self.opening_gap_guard < 1:
            raise ValueError("Invalid execution configuration")


def stamp_duty(date):
    return .0005 if pd.Timestamp(date) >= pd.Timestamp("2023-08-28") else .001


def simulate_cn(signals, quotes, calendar, config, start, end):
    """One open rebalance/day; sells use only units carried from prior sessions.

    Quotes use adjusted units; commissions/slippage are notional costs. No lots,
    minimum commissions or liquidity model. Guard is a conservative assumption,
    NOT a reconstruction of actual exchange price limits or ST eligibility.
    """
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar must be unique and increasing")
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(sessions) < 2 or sessions[0] == calendar[0]:
        raise ValueError("Need at least two sessions and a prior signal session")
    if quotes.duplicated(["date", "ticker"]).any() or signals.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate quotes/signals")
    qdates = {d: g.set_index("ticker") for d, g in quotes.groupby("date", sort=False)}
    sdates = {d: g.set_index("ticker")["prediction"] for d, g in signals.groupby("date", sort=False)}
    prior = dict(zip(calendar[1:], calendar[:-1]))
    cash, previous_nav, benchmark_nav = 1., 1., 1.
    units, marks, bought_on = {}, {}, {}
    ledger, daily, positions = [], [], []
    for day in sessions:
        source_day = prior[day]
        current = qdates.get(day, pd.DataFrame(columns=["open", "close"]))
        yesterday = qdates.get(source_day, pd.DataFrame(columns=["open", "close"]))

        def price(table, ticker, field):
            value = table.at[ticker, field] if ticker in table.index else np.nan
            return float(value) if np.isfinite(value) and value > 0 else np.nan

        opening = {ticker: price(current, ticker, "open") for ticker in units}
        nav_open = cash + sum(n * (opening[t] if np.isfinite(opening[t]) else marks[t]) for t, n in units.items())
        signal = sdates.get(source_day, pd.Series(dtype=float))
        ranking = signal[np.isfinite(signal)].rename("prediction").reset_index()
        if not ranking.empty:
            ranking.columns = ["ticker", "prediction"]
            ranking = ranking.sort_values(["prediction", "ticker"], ascending=[False, True])
        chosen = [] if day == sessions[-1] else ranking.head(config.topk)["ticker"].tolist() if not ranking.empty else []
        targets = {ticker: nav_open * config.exposure / config.topk for ticker in chosen}
        cost = traded = 0.
        blocked_buy = blocked_sell = 0

        def tradable(ticker, side):
            op, previous = price(current, ticker, "open"), price(yesterday, ticker, "close")
            if not np.isfinite(op) or not np.isfinite(previous):
                return False
            gap = op / previous - 1
            return gap < config.opening_gap_guard if side == "buy" else gap > -config.opening_gap_guard

        def trade(ticker, quantity, side):
            nonlocal cash, cost, traded
            op = price(current, ticker, "open")
            notional = quantity * op
            tax = notional * stamp_duty(day) if side == "sell" else 0.
            fees = notional * (config.commission + config.slippage) + tax
            if side == "sell":
                if bought_on[ticker] >= day:
                    raise ValueError("T+1 violation")
                cash += notional - fees
                units[ticker] -= quantity
                if units[ticker] <= 1e-12:
                    del units[ticker]
            else:
                cash -= notional + fees
                units[ticker] = units.get(ticker, 0.) + quantity
                marks[ticker], bought_on[ticker] = op, day
            cost += fees
            traded += notional
            ledger.append({"date": day, "signal_date": source_day, "ticker": ticker, "side": side,
                           "adjusted_units": quantity, "open_mid": op, "notional_nav_units": notional,
                           "commission": notional * config.commission, "slippage": notional * config.slippage,
                           "stamp_duty": tax, "total_cost": fees})

        # All sell quantities existed before this open; no within-day resale.
        for ticker in list(units):
            op = price(current, ticker, "open")
            if not np.isfinite(op) or not tradable(ticker, "sell"):
                blocked_sell += 1
                continue
            quantity = max(0., units[ticker] - targets.get(ticker, 0.) / op)
            if quantity > 1e-12:
                trade(ticker, quantity, "sell")
        requests = {}
        for ticker, target in targets.items():
            if not tradable(ticker, "buy"):
                blocked_buy += 1
                continue
            op = price(current, ticker, "open")
            needed = max(0., target - units.get(ticker, 0.) * op)
            if needed > 1e-12:
                requests[ticker] = needed
        funding = sum(requests.values()) * (1 + config.commission + config.slippage)
        scale = min(1., max(cash, 0.) / funding) if funding else 0.
        for ticker, notional in requests.items():
            trade(ticker, notional * scale / price(current, ticker, "open"), "buy")
        if cash < -1e-10:
            raise ValueError("Negative cash")
        cash = max(0., cash)
        stale = 0
        for ticker, quantity in units.items():
            close = price(current, ticker, "close")
            if np.isfinite(close):
                marks[ticker] = close
            else:
                op = price(current, ticker, "open")
                if np.isfinite(op):
                    marks[ticker] = op
                stale += 1
            positions.append({"date": day, "ticker": ticker, "adjusted_units": quantity,
                              "mark": marks[ticker], "stale_mark": not np.isfinite(close)})
        nav = cash + sum(n * marks[t] for t, n in units.items())
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Invalid NAV")
        index_op, index_cl = price(current, "SH000300", "open"), price(current, "SH000300", "close")
        prev_index = price(yesterday, "SH000300", "close")
        if day == sessions[0]:
            br = index_cl / index_op - 1
        elif day == sessions[-1]:
            br = index_op / prev_index - 1
        else:
            br = index_cl / prev_index - 1
        if not np.isfinite(br):
            raise ValueError("Missing CSI300 benchmark price")
        benchmark_nav *= 1 + br
        net = nav / previous_nav - 1
        daily.append({"date": day, "net_return": net, "gross_return": net + cost / previous_nav,
                      "benchmark_return": br, "turnover": traded / previous_nav / 2,
                      "transaction_cost": cost / previous_nav, "nav": nav, "cash": cash,
                      "held_assets": len(units), "stale_assets": stale, "blocked_buys": blocked_buy,
                      "blocked_sells": blocked_sell, "benchmark_nav": benchmark_nav})
        previous_nav = nav
    report = pd.DataFrame(daily).set_index("date")
    audit = {"liquidation_complete": not units, "ending_positions": len(units),
             "stale_valuation_days": int(report.stale_assets.gt(0).sum()),
             "blocked_buys": int(report.blocked_buys.sum()), "blocked_sells": int(report.blocked_sells.sum()),
             "config": asdict(config), "fractional_adjusted_units": True}
    trade_columns = ["date", "signal_date", "ticker", "side", "adjusted_units", "open_mid", "notional_nav_units",
                     "commission", "slippage", "stamp_duty", "total_cost"]
    position_columns = ["date", "ticker", "adjusted_units", "mark", "stale_mark"]
    return report, pd.DataFrame(ledger, columns=trade_columns), pd.DataFrame(positions, columns=position_columns), audit
