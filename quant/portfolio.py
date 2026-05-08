from __future__ import annotations
import numpy as np
import pandas as pd


def build_portfolio_from_scores(
    scores: pd.Series,
    mode: str = "long_only",
    long_frac: float = 0.2,
    short_frac: float = 0.2,
    gross_leverage: float = 1.0,
    net_exposure: float | None = None,
    dollar_neutral: bool = True,
    rebalance_freq: int = 1,
) -> pd.DataFrame:
    """
    Rank-based portfolio per date.
    Output weights: [date x symbol]
    """
    if mode not in {"long_only", "long_short"}:
        raise ValueError(f"Unsupported portfolio mode {mode}")
    if gross_leverage <= 0:
        raise ValueError("gross_leverage must be positive")
    if rebalance_freq < 1:
        raise ValueError("rebalance_freq must be >= 1")
    if net_exposure is not None and abs(net_exposure) > gross_leverage:
        raise ValueError("abs(net_exposure) cannot exceed gross_leverage")

    df = scores.rename("score").reset_index()
    df.columns = ["date", "symbol", "score"]
    weights = []

    def rank_weights(symbols, total_weight: float) -> pd.Series:
        ranks = pd.Series(range(len(symbols), 0, -1), index=symbols, dtype=float)
        return ranks / ranks.sum() * total_weight

    for i, (d, g) in enumerate(df.groupby("date")):
        if i % rebalance_freq != 0:
            continue
        g = g.dropna()
        n = len(g)
        if n < 5:
            continue
        kL = max(1, int(np.floor(n * long_frac)))
        kS = max(1, int(np.floor(n * short_frac))) if mode == "long_short" else 0

        g = g.sort_values("score", ascending=False)
        long_syms = g.head(kL)["symbol"].tolist()
        short_syms = g.tail(kS)["symbol"].tolist()

        w = pd.Series(0.0, index=g["symbol"].values)
        if mode == "long_only":
            w.loc[long_syms] = rank_weights(long_syms, gross_leverage)
        else:
            target_net = 0.0 if dollar_neutral else (net_exposure or 0.0)
            long_budget = 0.5 * (gross_leverage + target_net)
            short_budget = 0.5 * (gross_leverage - target_net)
            w.loc[long_syms] = rank_weights(long_syms, long_budget)
            if short_budget > 0:
                w.loc[short_syms] = -rank_weights(list(reversed(short_syms)), short_budget)

        gross = w.abs().sum()
        if gross > 0 and abs(gross - gross_leverage) > 1e-10:
            w = w / gross * gross_leverage

        w.name = d
        weights.append(w)

    W = pd.DataFrame(weights).sort_index()
    W.index.name = "date"
    return W
