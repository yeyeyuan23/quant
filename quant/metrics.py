from __future__ import annotations
import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_performance(
    equity_curve: pd.Series,
    daily_pnl: pd.Series,
    turnover: pd.Series | None = None,
    ann_factor: int = 252,
) -> pd.Series:
    r = daily_pnl.dropna()
    if len(r) < 10:
        raise ValueError("Not enough returns to summarize.")

    cagr = float(equity_curve.iloc[-1] ** (ann_factor / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(ann_factor))
    sharpe = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(ann_factor))
    mdd = _max_drawdown(equity_curve)
    summary = {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "TotalReturn": float(equity_curve.iloc[-1] - 1.0),
        "NumDays": int(len(r)),
    }
    if turnover is not None and len(turnover) > 0:
        summary["AvgTurnover"] = float(turnover.mean())

    return pd.Series(summary)


def compare_to_benchmarks(
    prices: pd.DataFrame,
    strategy_pnl: pd.Series,
    strategy_equity: pd.Series,
    benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ"),
) -> pd.DataFrame:
    rets = prices.pct_change().fillna(0.0).reindex(strategy_pnl.index)
    rows = {"Strategy": summarize_performance(strategy_equity, strategy_pnl)}

    for sym in benchmark_symbols:
        if sym in rets:
            pnl = rets[sym]
            rows[sym] = summarize_performance((1.0 + pnl).cumprod(), pnl)

    equal_weight = rets.mean(axis=1)
    rows["EqualWeight"] = summarize_performance(
        (1.0 + equal_weight).cumprod(), equal_weight
    )

    return pd.DataFrame(rows).T
