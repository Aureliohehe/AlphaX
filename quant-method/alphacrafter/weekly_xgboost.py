"""Frozen XGBoost scorer for the CSI300 weekly four-factor comparison."""

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


def fit_weekly_scorer(frame, columns, first_signal, end):
    """Fit on matured training labels and score the comparison window once."""
    import xgboost as xgb

    maturity = frame["label_end_5d"]
    training = frame.loc[frame.date.between(TRAIN_START, TRAIN_END) & maturity.le(TRAIN_END)]
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
        "training_window": [TRAIN_START, TRAIN_END],
        "last_label_end": str(maturity.loc[labels.index].max().date()),
        "training_rows": len(labels),
        "target": TARGET,
        "parameters": MODEL_PARAMS,
        "xgboost_version": xgb.__version__,
    }
    return prediction, model, metadata
