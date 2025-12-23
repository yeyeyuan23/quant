from __future__ import annotations
import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_performance(
    equity_curve: pd.Series, daily_pnl: pd.Series, ann_factor: int = 252
) -> pd.Series:
    r = daily_pnl.dropna()
    if len(r) < 10:
        raise ValueError("Not enough returns to summarize.")

    cagr = float(equity_curve.iloc[-1] ** (ann_factor / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(ann_factor))
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(ann_factor))
    mdd = _max_drawdown(equity_curve)
    avg_turnover = float(
        r.index.to_series().map(lambda d: 1).mean()
    )  # placeholder if needed

    return pd.Series(
        {
            "CAGR": cagr,
            "AnnVol": vol,
            "Sharpe": sharpe,
            "MaxDrawdown": mdd,
            "TotalReturn": float(equity_curve.iloc[-1] - 1.0),
            "NumDays": int(len(r)),
        }
    )
