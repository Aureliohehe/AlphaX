"""Frozen XGBoost scorer for the CSI300 weekly four-factor comparison."""

from dataclasses import replace

import numpy as np
import pandas as pd


TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
TARGET = "target_open_open_5d"
MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": .03,
    "min_child_weight": 20,
    "subsample": .70,
    "colsample_bytree": .75,
    "gamma": .10,
    "reg_alpha": 0.,
    "reg_lambda": 10.,
    "objective": "reg:pseudohubererror",
    "n_estimators": 25,
    "random_state": 42,
    "n_jobs": 4,
}


def fit_weekly_scorer(frame, columns, first_signal, end, *, training_window=None):
    """Fit on matured training labels and score the comparison window once."""
    import xgboost as xgb

    train_start, train_end = map(pd.Timestamp, training_window or (TRAIN_START, TRAIN_END))
    if not train_start <= train_end <= pd.Timestamp(first_signal):
        raise ValueError("XGBoost training must end by the first signal")
    maturity = frame["label_end_5d"]
    training = frame.loc[frame.date.between(train_start, train_end) & maturity.le(train_end)]
    inputs = (training.groupby("date")[columns].rank(pct=True) - .5) * 3.46
    complete = training.loc[inputs.notna().all(axis=1)]
    labels = ((complete.groupby("date")[TARGET].rank(pct=True) - .5) * 3.46).dropna()
    if labels.empty:
        raise ValueError("No complete 5-session training labels for the weekly XGBoost scorer")

    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(inputs.loc[labels.index].astype("float32"), labels.astype("float32"))

    scoring = frame.loc[frame.date.between(first_signal, end)]
    ranked = ((scoring.groupby("date")[columns].rank(pct=True) - .5) * 3.46).dropna()
    if ranked.empty:
        raise ValueError("No complete four-factor scores in the XGBoost comparison window")
    prediction = pd.Series(model.predict(ranked.astype("float32")), index=ranked.index)
    metadata = {
        "training_window": [str(train_start.date()), str(train_end.date())],
        "last_label_end": str(maturity.loc[labels.index].max().date()),
        "training_rows": len(labels),
        "target": TARGET,
        "parameters": MODEL_PARAMS,
        "xgboost_version": xgb.__version__,
    }
    return prediction, model, metadata


def residual_factor_rank(sample, base_columns, candidate_column):
    """Rank only the candidate variation unexplained by core ranks on the signal date."""
    ranks = sample.groupby("date")[[*base_columns, candidate_column]].rank(pct=True)
    residuals = pd.Series(index=sample.index, dtype=float)
    for _, indices in sample.groupby("date").groups.items():
        block = ranks.loc[indices]
        core = block[base_columns].to_numpy()
        design = np.column_stack((np.ones(len(core)), core))
        candidate = block[candidate_column].to_numpy()
        coefficients = np.linalg.lstsq(design, candidate, rcond=None)[0]
        residuals.loc[indices] = np.round(candidate - design @ coefficients, 12)
    return residuals


def evaluate_factor_additions(frame, score_columns, core, variants, calendar, quotes, config, window,
                              *, training_window=None, augmentation_weight=1., augmentation_method="refit",
                              augmentation_active=None):
    """Compare one additional Miner input at a time on identical complete rows."""
    from admission import assess_weekly_substitution
    from cn_portfolio import simulate_cn
    from medium_frequency import anchor_mask
    from trader_harness import AUDIT_CHECKS
    from trader_lite import summarize

    start, end = window
    if not 0 < augmentation_weight <= 1:
        raise ValueError("XGBoost augmentation weight must be greater than zero and at most one")
    if augmentation_method not in {"refit", "factor_rank", "residual_rank"}:
        raise ValueError("Unknown XGBoost augmentation method")
    if augmentation_active is not None and (not augmentation_active.index.is_unique
                                               or not pd.api.types.is_bool_dtype(augmentation_active)):
        raise ValueError("Augmentation activation requires one boolean per signal date")
    train_start, train_end = map(pd.Timestamp, training_window or (TRAIN_START, TRAIN_END))
    if not train_start <= train_end < pd.Timestamp(start):
        raise ValueError("XGBoost addition evaluation must start after its label cutoff")
    first = calendar[calendar >= start][0]
    previous = calendar[calendar.get_loc(first) - 1]
    anchors = anchor_mask(frame.date, calendar, first, 5) & frame.date.between(previous, end)
    configs = {"1": config, "2": replace(config, commission=2 * config.commission, slippage=2 * config.slippage)}
    base_columns = [score_columns[name] for name in core]
    train_mask = frame.date.between(train_start, train_end) & frame.label_end_5d.le(train_end) & frame[TARGET].notna()
    training = frame.loc[train_mask]
    base_values = training[base_columns].where(np.isfinite(training[base_columns]))
    ranked = base_values.groupby(training.date).rank(pct=True)
    seen = {name: ranked[score_columns[name]] for name in core}
    for candidate in variants:
        name = candidate["candidate_id"]
        column = score_columns[name]
        entry = {"id": name, "control": candidate.get("control_id"), "expression": candidate["expression_text"]}
        values = training[column].where(np.isfinite(training[column]))
        ranks = values.groupby(training.date).rank(pct=True)
        duplicate = next((name for name, other in seen.items() if ranks.equals(other)), None)
        if duplicate is not None:
            yield {**entry, "status": "rank_duplicate_in_training", "equivalent_to": duplicate,
                   "qualifies_in_replay": False}
            continue
        columns = [*base_columns, column]
        common = frame.loc[np.isfinite(frame[columns]).all(axis=1)]
        sample = common.loc[anchors.loc[common.index], ["date", "ticker"]]
        if sample.empty or not train_mask.loc[common.index].any():
            yield {**entry, "status": "missing_complete_samples", "qualifies_in_replay": False}
            continue
        seen[name] = ranks
        active = (sample.date.map(augmentation_active).eq(True) if augmentation_active is not None
                  else pd.Series(True, index=sample.index))
        models = {}
        baseline_scores = None
        for model_name, features in (("core", base_columns), ("augmented", columns)):
            if model_name == "augmented" and augmentation_method != "refit":
                if augmentation_method == "residual_rank":
                    prediction = residual_factor_rank(common.loc[sample.index], base_columns, column)
                else:
                    prediction = common.loc[sample.index, column]
                metadata = models["core"]["training"]
            else:
                prediction, _, metadata = fit_weekly_scorer(common, features, previous, end,
                                                            training_window=training_window)
            prediction = prediction.reindex(sample.index)
            if model_name == "core":
                baseline_scores = prediction
            elif augmentation_weight < 1 or augmentation_method != "refit":
                core_rank = baseline_scores.groupby(sample.date).rank(pct=True)
                addition_rank = prediction.groupby(sample.date).rank(pct=True)
                prediction = (1 - augmentation_weight) * core_rank + augmentation_weight * addition_rank
            if model_name == "augmented":
                prediction = prediction.where(active, baseline_scores)
            signals = sample.assign(prediction=prediction)
            costs = {}
            for cost, execution in configs.items():
                report, _, _, audit = simulate_cn(signals, quotes, calendar, execution, start, end)
                costs[cost] = {
                    **summarize(report, 1., annual_risk_free_rate=0.),
                    "calendar_year_returns": {str(year): float((1 + rows.net_return).prod() - 1)
                                              for year, rows in report.groupby(report.index.year)},
                    "execution_passes": all(audit.get(check) for check in AUDIT_CHECKS),
                }
            models[model_name] = {"training": metadata, "costs": costs}
        baseline, augmented = (models[model]["costs"] for model in ("core", "augmented"))
        if models["core"]["training"]["training_rows"] != models["augmented"]["training"]["training_rows"]:
            raise ValueError("XGBoost addition models did not train on matching rows")
        checks = assess_weekly_substitution(baseline, augmented, independent=False, require_each_year=True)
        passes = all(trial["execution_passes"] for runs in (baseline, augmented) for trial in runs.values())
        yield {**entry, "status": "evaluated",
               "augmentation_weight": augmentation_weight,
               "augmentation_method": augmentation_method,
               "active_signal_dates": int(sample.loc[active, "date"].nunique()),
               "total_signal_dates": int(sample.date.nunique()),
               "signal_rows": len(sample), "results": models, "robustness": checks, "execution_passes": passes,
               "qualifies_in_replay": passes and checks["passes_paired_checks"]}
