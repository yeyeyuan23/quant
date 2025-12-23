from __future__ import annotations
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def walkforward_train_predict(
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict,
    train_window: int,
    retrain_freq: int,
    model_name: str = "ridge",
    alpha: float = 10.0,
) -> pd.Series:
    """
    Walk-forward training on dates. For each date t, train on (t-train_window ... t-1),
    predict scores for date t (cross-section).
    Return: scores Series indexed by (date, symbol)
    """
    dates = meta["dates"]
    scores = []

    # model
    if model_name == "ridge":
        mdl = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=alpha, random_state=42)),
            ]
        )
    else:
        raise ValueError(f"Unsupported model {model_name}")

    last_fit_i = None
    for i, d in enumerate(dates):
        if i < 5:
            continue
        if i < train_window:
            continue

        # Retrain schedule
        if (last_fit_i is None) or ((i - last_fit_i) >= retrain_freq):
            train_dates = dates[i - train_window : i]
            tr_idx = X.index.get_level_values(0).isin(train_dates)
            X_tr = X.loc[tr_idx]
            y_tr = y.loc[tr_idx]
            mdl.fit(X_tr.values, y_tr.values)
            last_fit_i = i

        # Predict for date d
        te_idx = X.index.get_level_values(0) == d
        X_te = X.loc[te_idx]
        if len(X_te) == 0:
            continue
        s = mdl.predict(X_te.values)
        s = pd.Series(s, index=X_te.index, name="score")
        scores.append(s)

    if not scores:
        raise RuntimeError("No scores produced. Check date range / train_window.")
    return pd.concat(scores).sort_index()
