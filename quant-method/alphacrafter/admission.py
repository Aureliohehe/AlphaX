"""Research eligibility and portfolio admission are separate decisions."""
from dataclasses import asdict, dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class ResearchPolicy:
    min_rank_ic: float = .015
    min_rank_icir: float = .10
    min_hit: float = .55
    min_coverage: float = .90
    max_turnover: float = .40

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Research thresholds must be finite")
        if self.min_rank_ic < 0 or self.min_rank_icir < 0:
            raise ValueError("Research strength thresholds must be nonnegative")
        if not .5 <= self.min_hit < 1 or not 0 <= self.min_coverage <= 1 or not 0 < self.max_turnover <= 1:
            raise ValueError("Invalid research hit/coverage/turnover thresholds")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def research_failures(metric, direction, policy):
    checks = {}
    for key, floor in (("mean_daily_rank_ic", policy.min_rank_ic), ("rank_icir", policy.min_rank_icir)):
        value = metric.get(key)
        checks[key] = direction is not None and finite(value) and direction * value > floor
    hit = metric.get("rank_ic_hit_ratio")
    checks["hit_ratio"] = (direction is not None and finite(hit) and 0 <= hit <= 1
                           and (hit if direction > 0 else 1 - hit) > policy.min_hit)
    coverage, turnover = metric.get("coverage"), metric.get("turnover")
    checks["coverage"] = finite(coverage) and policy.min_coverage < coverage <= 1
    checks["turnover"] = finite(turnover) and 0 <= turnover < policy.max_turnover
    checks["observations"] = finite(metric.get("rows")) and metric["rows"] > 0
    return [name for name, passed in checks.items() if not passed]


def build_research_pool(results, definitions, audit, market="unknown", policy=None):
    """Read train/development only; direction is fit separately per horizon."""
    policy = policy or ResearchPolicy()
    if market not in ("us", "cn", "unknown"):
        raise ValueError("Unknown research market")
    windows = audit["windows"]
    train_start, train_end = map(pd.Timestamp, windows["train"])
    dev_start, dev_end = map(pd.Timestamp, windows["validation"])
    cutoff = pd.Timestamp(audit["as_of"])
    if any(pd.isna(d) for d in (train_start, train_end, dev_start, dev_end, cutoff)) or not train_start <= train_end < dev_start <= dev_end:
        raise ValueError("Research requires ordered train/development windows and a valid cutoff")
    complete = dev_end <= cutoff
    by_id = {r["candidate_id"]: r for r in definitions}
    if len(by_id) != len(definitions) or len({r["candidate_id"] for r in results}) != len(results):
        raise ValueError("Duplicate candidate IDs")
    if set(by_id) != {r["candidate_id"] for r in results}:
        raise ValueError("Research definitions and metric candidate sets differ")
    selected = {"1": [], "5": []}
    assessments = []
    wanted = ("rows", "mean_daily_rank_ic", "rank_icir", "rank_ic_hit_ratio", "coverage", "turnover")
    for row in results:
        for h in ("1", "5"):
            values = row.get("horizons", {}).get(h, {})
            metrics = {name: {k: values.get(name, {}).get(k) for k in wanted} for name in ("train", "validation")}
            fit = metrics["train"]["mean_daily_rank_ic"]
            direction = (1 if fit > 0 else -1) if finite(fit) and fit != 0 else None
            failures = {name: research_failures(m, direction, policy) for name, m in metrics.items()}
            if not complete:
                failures["window_completion"] = ["development_incomplete"]
            eligible = not any(failures.values())
            metrics = {name: {k: v if finite(v) else None for k, v in m.items()} for name, m in metrics.items()}
            assessments.append({"candidate_id": row["candidate_id"], "horizon": int(h), "direction": direction,
                                "research_eligible": eligible, "metrics": metrics, "failed_checks": failures})
            if eligible:
                definition = by_id[row["candidate_id"]]
                selected[h].append({"candidate_id": row["candidate_id"], "horizon": int(h), "direction": direction,
                                    "status": "research_candidate", "strategy_admitted": False,
                                    "definition": definition, "metrics": metrics})
    pool = {"artifact_type": "research_candidate_pool_v1", "admission_stage": "research",
            "status": "assessed" if complete else "pending_complete_development",
            "scope": "research_only_not_a_trading_factor_library", "market": market,
            "label_mode": audit.get("label_mode", "unknown"), "as_of": str(cutoff.date()),
            "research_end": str(dev_end.date()),
            "windows": {"train": windows["train"], "development": windows["validation"]},
            "policy": asdict(policy), "direction_policy": "fit_on_training_rank_ic_separately_per_horizon",
            "evaluated_candidates": len(results), "candidates_by_horizon": selected,
            "eligible_counts": {h: len(v) for h, v in selected.items()},
            "strategy_admitted_count": 0}
    return pool, assessments


def strategy_plan(pool, validation_start, validation_end, holdout_start, holdout_end):
    if pool.get("artifact_type") != "research_candidate_pool_v1":
        raise ValueError("Expected a research candidate pool")
    research_cutoff = max(pd.Timestamp(pool["research_end"]), pd.Timestamp(pool["as_of"]))
    dates = list(map(pd.Timestamp, (research_cutoff, validation_start, validation_end, holdout_start, holdout_end)))
    if any(pd.isna(d) for d in dates) or not dates[0] < dates[1] <= dates[2] < dates[3] <= dates[4]:
        raise ValueError("Strategy validation must follow research and precede holdout")
    return {"artifact_type": "strategy_validation_plan_v1", "admission_stage": "strategy",
            "status": "pending_strategy_definition", "strategy_admitted": False,
            "market": pool["market"], "research_end": pool["research_end"],
            "validation": [str(dates[1].date()), str(dates[2].date())],
            "holdout": [str(dates[3].date()), str(dates[4].date())],
            "selection_constraints": {"annualized_return": ">0.08", "sharpe": ">0.6", "max_drawdown": ">-0.08"},
            "return_basis": "net_of_execution_costs", "max_validation_trials": 3,
            "factor_gate": "research_eligibility; original_joint_factor_gate_not_required",
            "freeze_before_validation": ["factor_ids_and_horizon", "directions_and_weights", "portfolio_config", "execution_and_costs"],
            "execution_requirement": ("A-share multi-session simulator required; existing intraday Trader is incompatible"
                                      if pool["market"] == "cn" else "Execution model must match research horizon and labels")}


def validate_trader_pool(pool, horizon, validation_start):
    """The existing Trader supports US execution-aligned 1d experiments only."""
    if pool.get("artifact_type") != "research_candidate_pool_v1":
        raise ValueError("Expected a research candidate pool")
    if pool.get("market") != "us" or pool.get("label_mode") != "execution" or horizon != 1:
        raise ValueError("Research market/labels/horizon require a different execution simulator; intraday Trader supports US execution 1d only")
    cutoff = max(pd.Timestamp(pool["research_end"]), pd.Timestamp(pool["as_of"]))
    if pd.isna(cutoff) or pd.Timestamp(validation_start) <= cutoff:
        raise ValueError("Independent strategy validation must start after research cutoff")
    if not pool.get("candidates_by_horizon", {}).get(str(horizon)):
        raise ValueError("Research pool has no eligible candidates for this horizon")
