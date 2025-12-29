from __future__ import annotations
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    commission_bps: float = 0.5,
    slippage_bps: float = 1.0,
):
    """
    Daily close-to-close backtest.
    PnL(t) = sum_i w_{t-1,i} * r_{t,i} - cost(turnover)
    turnover = sum_i |w_t - w_{t-1}|
    costs in bps applied to turnover.
    """
    px = prices.sort_index().ffill()
    rets = px.pct_change().fillna(0.0)

    # align dates
    W = weights.reindex(px.index).ffill().fillna(0.0)

    # trim warmup period with no active positions
    active = W.abs().sum(axis=1) > 0
    if active.any():
        start_date = active.idxmax()
        px = px.loc[start_date:]
        rets = rets.loc[start_date:]
        W = W.loc[start_date:]

    # turnover
    dW = W.diff().abs()
    turnover = dW.sum(axis=1).fillna(0.0)
    # align trading costs with the weights used for the return (t-1 -> t)
    turnover_for_costs = turnover.shift(1).fillna(0.0)

    cost_rate = (commission_bps + slippage_bps) / 10000.0
    costs = turnover_for_costs * cost_rate

    # use yesterday weights for today's return
    pnl_gross = (W.shift(1).fillna(0.0) * rets).sum(axis=1)
    daily_pnl = pnl_gross - costs

    equity_curve = (1.0 + daily_pnl).cumprod()
    return {
        "daily_pnl": daily_pnl,
        "equity_curve": equity_curve,
        "turnover": turnover,
        "weights": W,
        "costs": costs,
    }
