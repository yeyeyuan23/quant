from __future__ import annotations
import numpy as np
import pandas as pd


def build_portfolio_from_scores(
    scores: pd.Series,
    long_frac: float = 0.2,
    short_frac: float = 0.2,
    gross_leverage: float = 1.0,
    dollar_neutral: bool = True,
) -> pd.DataFrame:
    """
    Rank-based long/short portfolio per date.
    Output weights: [date x symbol]
    """
    df = scores.rename("score").reset_index()
    df.columns = ["date", "symbol", "score"]
    weights = []

    for d, g in df.groupby("date"):
        g = g.dropna()
        n = len(g)
        if n < 5:
            continue
        kL = max(1, int(np.floor(n * long_frac)))
        kS = max(1, int(np.floor(n * short_frac)))

        g = g.sort_values("score", ascending=False)
        long_syms = g.head(kL)["symbol"].tolist()
        short_syms = g.tail(kS)["symbol"].tolist()

        w = pd.Series(0.0, index=g["symbol"].values)
        if kL > 0:
            w.loc[long_syms] = 0.5 * gross_leverage / kL
        if kS > 0:
            w.loc[short_syms] = -0.5 * gross_leverage / kS

        if not dollar_neutral:
            # rescale to gross leverage only
            gross = w.abs().sum()
            if gross > 0:
                w = w / gross * gross_leverage
        else:
            # ensure exact neutrality (small numerical fix)
            w = w - w.mean()

            gross = w.abs().sum()
            if gross > 0:
                w = w / gross * gross_leverage

        w.name = d
        weights.append(w)

    W = pd.DataFrame(weights).sort_index()
    W.index.name = "date"
    return W
