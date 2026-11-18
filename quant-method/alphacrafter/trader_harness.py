"""Select a China A-share execution frequency on a common validation window."""
from cn_portfolio import CNConfig
from medium_frequency import strategy_presets
from trader_lite import is_feasible, select_configuration


def frequency_presets():
    return {
        1: CNConfig(30, rebalance_interval=1, min_holding_sessions=1, sell_rank_buffer=30,
                    max_daily_turnover=.10, max_daily_cost=.0003, rebalance_existing=False),
        **strategy_presets(),
    }


def select_frequency(trials, window, *, evaluated):
    """Compare complete cost-adjusted strategies, with cash as the fallback."""
    horizons = [trial["horizon"] for trial in trials]
    if evaluated and sorted(horizons) != sorted(frequency_presets()):
        raise ValueError("Frequency comparison requires one trial record for each preset")
    eligible = []
    for trial in trials:
        if trial["status"] != "feasible_in_proxy":
            continue
        config = trial["config"]
        if config["rebalance_interval"] != trial["horizon"]:
            raise ValueError("Trial frequency differs from its execution configuration")
        if not all(trial["execution_audit"].get(key) for key in
                   ("liquidation_complete", "turnover_budget_respected", "cost_budget_respected")):
            continue
        if is_feasible(trial):
            eligible.append(trial)
    # Order resolves an exact Sharpe tie in favor of the lower-frequency strategy.
    eligible.sort(key=lambda trial: trial["horizon"], reverse=True)
    selected = select_configuration(eligible) if eligible else None
    return {
        "selected_horizon": selected["horizon"] if selected else None,
        "selected_config": selected["config"] if selected else None,
        "selection_window": list(window),
        "selection_rule": "highest net daily Sharpe among feasible trials; ties favor slower trading",
        "risk_thresholds": {"annualized_return_gt": .08, "sharpe_gt": .60, "max_drawdown_gt": -.08},
        "status": ("selected_in_iteration_proxy" if selected else "no_feasible_strategy"
                   if evaluated else "not_evaluated"),
        "holdout_evaluated": False,
    }
