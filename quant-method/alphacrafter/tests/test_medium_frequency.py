import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from admission import build_research_pool
from cn_portfolio import CNConfig, simulate_cn
from medium_frequency import (anchor_mask, definitions, execution_labels, factor_panel,
                              screen_signals, strategy_presets)
from temporal import is_feature_column


class MediumFrequencyTests(unittest.TestCase):
    def test_daily_target_fill_preserves_selection_budget_and_holding_locks(self):
        calendar = pd.bdate_range("2022-01-03", periods=100)
        quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100., "close": 100.}
                               for d in calendar for t in ("A", "B", "SH000300")])
        # Only the initial scheduled decision selects A; intermediate signals
        # request B, which must never be selected by the daily fill mechanism.
        signals = pd.DataFrame([{"date": d, "ticker": t, "prediction": float(t == ("A" if i % 20 == 0 else "B"))}
                                for i, d in enumerate(calendar) for t in ("A", "B")])
        config = replace(strategy_presets()[20], topk=1, fill_target_daily=True)
        report, trades, _, audit = simulate_cn(signals, quotes, calendar, config, calendar[1], calendar[-1])
        initial = trades[trades.date < calendar[21]]
        self.assertEqual(set(initial.ticker), {"A"})
        self.assertGreater(len(initial), 1)
        self.assertLess(report.cash.iloc[5] / report.nav.iloc[5], .21)
        self.assertTrue(audit["turnover_budget_respected"] and audit["cost_budget_respected"])
        self.assertTrue(audit["liquidation_complete"])
        last_buy = {}
        for row in trades.itertuples():
            if row.side == "buy":
                last_buy[row.ticker] = calendar.get_loc(row.date)
            else:
                self.assertGreaterEqual(calendar.get_loc(row.date) - last_buy[row.ticker], 20)
        changed = quotes.copy()
        changed.loc[changed.date >= calendar[3], "close"] *= 1.5
        _, other, _, _ = simulate_cn(signals, changed, calendar, config, calendar[1], calendar[-1])
        pd.testing.assert_frame_equal(trades[trades.date <= calendar[3]].reset_index(drop=True),
                                      other[other.date <= calendar[3]].reset_index(drop=True))

    def test_exposure_matched_index_equals_identical_asset_including_costs(self):
        calendar = pd.bdate_range("2022-01-03", periods=100)
        quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100 * 1.001**i,
                                "close": 100 * 1.001**i * (1.002 if i % 2 else .999)}
                               for i, d in enumerate(calendar) for t in ("A", "SH000300")])
        signals = pd.DataFrame({"date": calendar, "ticker": "A", "prediction": 1.})
        for mode in (False, True):
            config = replace(strategy_presets()[5], topk=1, fill_target_daily=mode)
            report, _, _, audit = simulate_cn(signals, quotes, calendar, config, calendar[1], calendar[-1])
            np.testing.assert_allclose(report.net_return, report.matched_index_net_return, atol=1e-12)
            self.assertTrue(audit["liquidation_complete"])

    def test_execution_labels_follow_market_sessions_and_skip_signal_close(self):
        dates = pd.Series(pd.bdate_range("2020-01-01", periods=30))
        opens = pd.Series(np.arange(30.) + 100)
        labels = execution_labels(dates, opens, pd.DatetimeIndex(dates))
        self.assertAlmostEqual(labels.loc[0, "target_open_open_5d"], 106 / 101 - 1)
        self.assertAlmostEqual(labels.loc[0, "target_open_open_20d"], 121 / 101 - 1)
        self.assertEqual(labels.loc[0, "label_end_20d"], dates.iloc[21])
        self.assertFalse(is_feature_column("label_start_20d"))
        opens.iloc[6] = np.nan
        labels = execution_labels(dates, opens, pd.DatetimeIndex(dates))
        self.assertTrue(pd.isna(labels.loc[0, "target_open_open_5d"]))
        self.assertEqual(labels.loc[0, "label_end_5d"], dates.iloc[6])

    def test_factor_values_use_only_past_and_missing_returns_are_not_filled(self):
        dates = pd.bdate_range("2018-01-01", periods=200)
        close = 100 + np.arange(200.) * .1 + np.sin(np.arange(200.))
        raw = pd.DataFrame({"date": dates, "ticker": "A", "open": close, "close": close,
                            "high": close + 1, "low": close - 1, "volume": 1000.})
        before = factor_panel(raw, dates)
        raw.loc[170:, ["open", "close", "high", "low"]] *= 2
        after = factor_panel(raw, dates)
        fields = [d["candidate_id"] for d in definitions()]
        pd.testing.assert_frame_equal(before.loc[:169, fields], after.loc[:169, fields])
        raw.loc[150, ["open", "close", "high", "low", "volume"]] = np.nan
        result = factor_panel(raw, dates)
        self.assertTrue(result.loc[150, fields].isna().all())
        self.assertTrue(pd.isna(result.loc[151, "mf_low_volatility_60"]))

    def test_research_pool_supports_independent_5_and_20_session_directions(self):
        metric = {"rows": 100, "mean_daily_rank_ic": .03, "rank_icir": .2,
                  "rank_ic_hit_ratio": .6, "coverage": .99, "turnover": .1}
        metrics = {str(h): {s: dict(metric) for s in ("train", "validation")} for h in (5, 20)}
        audit = {"as_of": "2021-12-31", "label_mode": "execution_open_open",
                 "windows": {"train": ["2010-01-01", "2018-12-31"], "validation": ["2019-01-01", "2021-12-31"]}}
        pool, _ = build_research_pool([{"candidate_id": "f", "horizons": metrics}],
                                     [{"candidate_id": "f", "features": ["f"]}], audit, "cn", horizons=(5, 20))
        self.assertEqual(pool["eligible_counts"], {"5": 1, "20": 1})
        self.assertEqual(pool["label_mode"], "execution_open_open")

    def test_screener_never_uses_not_yet_mature_labels(self):
        dates = pd.bdate_range("2020-01-01", periods=120)
        first_execution = dates[81]
        rows = []
        for index, date in enumerate(dates):
            for asset in range(5):
                rows.append({"date": date, "ticker": str(asset), "f": float(asset),
                             "target_open_open_5d": float(asset),
                             "label_end_5d": dates[index+6] if index+6 < len(dates) else pd.NaT})
        frame = pd.DataFrame(rows)
        pool = {"artifact_type": "research_candidate_pool_v1", "market": "cn", "label_mode": "execution_open_open",
                "as_of": str(dates[79].date()), "research_end": str(dates[79].date()), "candidates_by_horizon": {"5": [
            {"candidate_id": "f", "direction": 1, "definition": {"features": ["f"]}}]}}
        args = (pool, 5, dates, first_execution, dates[81], dates[-1])
        signals, selections = screen_signals(frame, *args, min_periods=2)
        self.assertFalse(signals.empty)
        self.assertTrue((pd.to_datetime(selections.last_label_end_used).dropna() <= selections.loc[selections.last_label_end_used.notna(), "date"]).all())
        changed = frame.copy()
        cutoff = dates[90]
        changed.loc[changed.label_end_5d > cutoff, "target_open_open_5d"] *= -100
        other, other_selections = screen_signals(changed, *args, min_periods=2)
        pd.testing.assert_frame_equal(signals[signals.date <= cutoff].reset_index(drop=True),
                                      other[other.date <= cutoff].reset_index(drop=True))
        pd.testing.assert_frame_equal(selections[selections.date <= cutoff].reset_index(drop=True),
                                      other_selections[other_selections.date <= cutoff].reset_index(drop=True))
        with self.assertRaisesRegex(ValueError, "research cutoff"):
            screen_signals(frame, pool, 5, dates, first_execution, dates[70], dates[-1], min_periods=2)

    def test_any_strength_screen_can_select_on_directional_hit_rate(self):
        dates = pd.bdate_range("2020-01-01", periods=36)
        frame = pd.DataFrame({"date": dates, "ticker": "x", "factor": 1.,
                              "target_open_open_1d": .01,
                              "label_end_1d": pd.Series(dates).shift(-2)})
        pool = {"artifact_type": "research_candidate_pool_v1", "market": "cn", "label_mode": "execution_open_open",
                "as_of": str(dates[14].date()), "research_end": str(dates[14].date()),
                "candidates_by_horizon": {"1": [{"candidate_id": "factor", "direction": 1,
                                                 "score_column": "factor"}]}}
        ic = pd.Series([.1, .1, .1, .1, -.9, -.9], index=dates[1:7])
        with patch("medium_frequency.daily_rank_ic", return_value=ic):
            strict, _ = screen_signals(frame, pool, 1, dates, dates[15], dates[15], dates[-1])
            relaxed, selections = screen_signals(
                frame, {**pool, "strength_rule": "any"}, 1, dates, dates[15], dates[15], dates[-1],
            )
        self.assertTrue(strict.empty)
        self.assertFalse(relaxed.empty)
        self.assertTrue((selections.last_label_end_used <= selections.date).all())

    def test_anchor_intervals_match_nonoverlapping_execution_periods(self):
        calendar = pd.bdate_range("2020-01-01", periods=70)
        dates = pd.Series(calendar)
        for h in (5, 20):
            anchors = np.flatnonzero(anchor_mask(dates, calendar, calendar[21], h))
            self.assertTrue(np.diff(anchors).tolist() == [h] * (len(anchors)-1))
            self.assertIn(20, anchors)

    def test_weekly_and_monthly_trade_budgets_holding_and_schedule(self):
        calendar = pd.bdate_range("2020-01-01", periods=130)
        quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100., "close": 100.}
                               for d in calendar for t in ("A", "B", "SH000300")])
        for h, preset in strategy_presets().items():
            config = CNConfig(1, rebalance_interval=h, min_holding_sessions=h,
                              max_daily_turnover=preset.max_daily_turnover,
                              max_daily_cost=preset.max_daily_cost, rebalance_existing=False)
            signals = pd.DataFrame([{"date": d, "ticker": t, "prediction": float((i//h)%2 == j)}
                                     for i, d in enumerate(calendar) for j, t in enumerate(("A", "B"))])
            report, trades, _, audit = simulate_cn(signals, quotes, calendar, config, calendar[1], calendar[-1])
            self.assertFalse(trades.empty)
            self.assertTrue(audit["turnover_budget_respected"])
            self.assertTrue(audit["cost_budget_respected"])
            self.assertTrue(audit["liquidation_complete"])
            position = {d: i for i, d in enumerate(calendar)}
            bought = {}
            for row in trades.itertuples():
                daily = report.loc[row.date]
                self.assertTrue(daily.scheduled_rebalance or daily.terminal_winddown)
                if row.side == "buy":
                    bought[row.ticker] = position[row.date]
                else:
                    self.assertGreaterEqual(position[row.date] - bought[row.ticker], h)

    def test_partial_initial_orders_can_fill_later_without_daily_rebalancing(self):
        calendar = pd.bdate_range("2020-01-01", periods=80)
        quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100., "close": 100.}
                               for d in calendar for t in ("A", "SH000300")])
        signals = pd.DataFrame({"date": calendar, "ticker": "A", "prediction": 1.})
        config = CNConfig(1, commission=0., slippage=0., rebalance_interval=5, min_holding_sessions=5,
                          max_daily_turnover=.1, rebalance_existing=False)
        report, trades, _, audit = simulate_cn(signals, quotes, calendar, config, calendar[1], calendar[-1])
        self.assertGreater((trades.side == "buy").sum(), 1)
        self.assertLess(report.cash.iloc[20], .21)
        self.assertTrue(audit["liquidation_complete"])

    def test_screen_to_portfolio_chain_runs_for_both_horizons(self):
        calendar = pd.bdate_range("2020-01-01", periods=200)
        frames, quotes = [], []
        for asset in range(5):
            prices = 100 * np.exp(asset * .001 * np.arange(len(calendar)))
            rows = pd.DataFrame({"date": calendar, "ticker": str(asset), "f": float(asset)})
            labels = execution_labels(rows.date, pd.Series(prices), calendar)
            frames.append(pd.concat([rows, labels], axis=1))
            quotes.append(pd.DataFrame({"date": calendar, "ticker": str(asset), "open": prices, "close": prices}))
        quotes.append(pd.DataFrame({"date": calendar, "ticker": "SH000300", "open": 100., "close": 100.}))
        frame = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        quotes = pd.concat(quotes, ignore_index=True)
        pool = {"artifact_type": "research_candidate_pool_v1", "market": "cn", "label_mode": "execution_open_open",
                "as_of": str(calendar[79].date()), "research_end": str(calendar[79].date()),
                "candidates_by_horizon": {str(h): [{"candidate_id": "f", "direction": 1,
                                                     "definition": {"features": ["f"]}}] for h in (5, 20)}}
        for h in (5, 20):
            signals, _ = screen_signals(frame, pool, h, calendar, calendar[81], calendar[81], calendar[-1], min_periods=2)
            config = CNConfig(1, rebalance_interval=h, min_holding_sessions=h, max_daily_turnover=.2,
                              max_daily_cost=.0005, rebalance_existing=False)
            report, trades, _, audit = simulate_cn(signals, quotes, calendar, config, calendar[81], calendar[-1])
            self.assertFalse(trades.empty)
            self.assertGreater(report.nav.iloc[-1], 1.)
            self.assertTrue(audit["liquidation_complete"])
            self.assertTrue(audit["cost_budget_respected"] and audit["turnover_budget_respected"])

    def test_rank_buffer_keeps_existing_stock_until_it_exits_retention_band(self):
        calendar = pd.bdate_range("2020-01-01", periods=65)
        quotes = pd.DataFrame([{"date": d, "ticker": t, "open": 100., "close": 100.}
                               for d in calendar for t in ("A", "B", "C", "SH000300")])
        rows = []
        for index, date in enumerate(calendar):
            ranking = {"A": 3, "B": 2, "C": 1} if index < 5 else {"B": 3, "A": 2, "C": 1} if index < 10 else {"B": 3, "C": 2, "A": 1}
            rows.extend({"date": date, "ticker": ticker, "prediction": score} for ticker, score in ranking.items())
        config = CNConfig(1, rebalance_interval=5, min_holding_sessions=5, sell_rank_buffer=1,
                          commission=0., slippage=0., rebalance_existing=False)
        _, trades, _, _ = simulate_cn(pd.DataFrame(rows), quotes, calendar, config, calendar[1], calendar[-1])
        self.assertTrue(trades[trades.date == calendar[6]].empty)
        self.assertEqual(trades[(trades.side == "sell") & (trades.ticker == "A")].date.iloc[0], calendar[11])


if __name__ == "__main__":
    unittest.main()
