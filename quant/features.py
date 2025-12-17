from __future__ import annotations
import numpy as np
import pandas as pd

def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional zscore per date."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)

def make_features_and_labels(
    prices: pd.DataFrame,
    lookback: int = 20,
    label_horizon: int = 1,
    min_history: int = 60
):
    """
    prices: [date x symbol]
    Return:
      X: MultiIndex (date, symbol) x features
      y: MultiIndex (date, symbol) next return
      meta: dict with dates index
    """
    px = prices.copy().sort_index().ffill()
    rets = px.pct_change()

    # Features (all cross-sectional z-scored for comparability)
    mom = (px / px.shift(lookback) - 1.0)                         # momentum
    mr  = -(px / px.shift(5) - 1.0)                               # mean-reversion (short horizon)
    vol = rets.rolling(lookback).std()                            # realized vol
    vol_chg = vol / vol.rolling(lookback).mean() - 1.0            # vol regime
    rsi_like = (rets.clip(lower=0).rolling(lookback).mean() /
                (rets.abs().rolling(lookback).mean() + 1e-12))    # proxy

    F = pd.concat({
        "mom": mom,
        "mr": mr,
        "vol": vol,
        "vol_chg": vol_chg,
        "rsi": rsi_like
    }, axis=1)

    # Cross-sectional normalize by date per feature
    feats = []
    for name in F.columns.levels[0]:
        feats.append(_zscore_cs(F[name]).rename(columns=lambda c: f"{name}"))
    # We want columns = feature names, but per symbol.
    # Build panel to long format: (date, symbol) -> feature vector
    feat_panel = pd.concat(feats, axis=1)

    # Label: forward return
    y = px.shift(-label_horizon) / px - 1.0

    # Drop early history & NaNs
    valid_dates = px.index[min_history: -label_horizon] if label_horizon > 0 else px.index[min_history:]
    feat_panel = feat_panel.loc[valid_dates]
    y = y.loc[valid_dates]

    X = feat_panel.stack().to_frame("value").unstack(level=0)
    # That reshape is messy; easier: build MultiIndex by stacking features separately:
    X = feat_panel.stack().rename_axis(["date", "symbol"]).reset_index()
    y_long = y.stack().rename("y").rename_axis(["date", "symbol"]).reset_index()

    df = X.merge(y_long, on=["date", "symbol"], how="inner")
    # Pivot features into columns
    X_wide = df.pivot_table(index=["date", "symbol"], values="value", columns="level_2")
    # Above column name depends on pandas; fix robustly:
    if X_wide.columns.name is None:
        # try alternative: rebuild directly
        raise RuntimeError("Unexpected pivot behavior. Please update pandas.")
    X_wide.columns = [str(c) for c in X_wide.columns]

    y_idx = df.set_index(["date", "symbol"])["y"].loc[X_wide.index]

    # Clean
    X_wide = X_wide.replace([np.inf, -np.inf], np.nan).dropna()
    y_idx = y_idx.loc[X_wide.index]

    meta = {"dates": sorted(X_wide.index.get_level_values(0).unique())}
    return X_wide, y_idx, meta