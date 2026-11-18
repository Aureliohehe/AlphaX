"""Select a China A-share execution frequency on a common validation window."""
import math

from cn_portfolio import CNConfig
from medium_frequency import strategy_presets
from trader_lite import is_feasible


def frequency_presets():
    return {
        1: CNConfig(30, rebalance_interval=1, min_holding_sessions=1, sell_rank_buffer=30,
                    max_daily_turnover=.10, max_daily_cost=.0003, rebalance_existing=False),
        **strategy_presets(),
    }


STANDARD_RISK = {"annualized_return": .08, "sharpe": .60, "max_drawdown": -.08}
WEEKLY_COMBINATION_RISK = {"annualized_return": .02, "sharpe": .25, "max_drawdown": -.12}
AUDIT_CHECKS = ("liquidation_complete", "turnover_budget_respected", "cost_budget_respected")


def requires_weekly_stress(horizon, members):
    return horizon == 5 and bool(members)


def strategy_is_feasible(trial):
    if not all(trial["execution_audit"].get(key) for key in AUDIT_CHECKS):
        return False
    if not requires_weekly_stress(trial["horizon"], trial.get("members")):
        return is_feasible(trial)
    stress = trial.get("double_cost")
    if not stress or not all(stress["execution_audit"].get(key) for key in AUDIT_CHECKS):
        return False
    for result in (trial, stress):
        for key, minimum in WEEKLY_COMBINATION_RISK.items():
            value = result.get(key)
            if value is None or not math.isfinite(value) or value <= minimum:
                return False
    return True


def select_frequency(trials, window, *, evaluated):
    """Compare complete cost-adjusted strategies, with cash as the fallback."""
    horizons = [trial["horizon"] for trial in trials]
    keys = {(trial["horizon"], trial.get("strategy_id")) for trial in trials}
    if evaluated and (set(horizons) != set(frequency_presets()) or len(keys) != len(trials)):
        raise ValueError("Frequency comparison requires coverage of every preset and unique trial IDs")
    eligible = []
    for trial in trials:
        if trial["status"] != "feasible_in_proxy":
            continue
        config = trial["config"]
        if config["rebalance_interval"] != trial["horizon"]:
            raise ValueError("Trial frequency differs from its execution configuration")
        if strategy_is_feasible(trial):
            eligible.append(trial)
    # Exact Sharpe ties favor slower trading, then the stable strategy ID.
    selected = min(eligible, key=lambda trial: (-trial["sharpe"], -trial["horizon"],
                                                trial.get("strategy_id") or ""), default=None)
    return {
        "selected_horizon": selected["horizon"] if selected else None,
        "selected_strategy_id": selected.get("strategy_id") if selected else None,
        "selected_members": selected.get("members") if selected else None,
        "selected_config": selected["config"] if selected else None,
        "selection_window": list(window),
        "selection_rule": "highest net daily Sharpe among feasible trials; ties favor slower trading",
        "risk_thresholds": {"standard": STANDARD_RISK,
                            "weekly_combination_both_costs": WEEKLY_COMBINATION_RISK},
        "status": ("selected_in_iteration_proxy" if selected else "no_feasible_strategy"
                   if evaluated else "not_evaluated"),
        "holdout_evaluated": False,
    }
