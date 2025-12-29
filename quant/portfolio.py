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
            long_scores = g.set_index("symbol").loc[long_syms, "score"].clip(lower=0.0)
            if long_scores.sum() <= 0:
                long_scores = pd.Series(1.0, index=long_scores.index)
            w.loc[long_syms] = long_scores / long_scores.sum() * 0.5 * gross_leverage
        if kS > 0:
            short_scores = (
                -g.set_index("symbol").loc[short_syms, "score"].clip(upper=0.0)
            )
            if short_scores.sum() > 0:
                w.loc[short_syms] = (
                    -short_scores / short_scores.sum() * 0.5 * gross_leverage
                )

        if not dollar_neutral:
            gross = w.abs().sum()
            if gross > 0:
                w = w / gross * gross_leverage
        elif w.abs().sum() > 0:
            if (w < 0).sum() == 0:
                gross = w.abs().sum()
                if gross > 0:
                    w = w / gross * gross_leverage
            else:
                w = w - w.mean()
                gross = w.abs().sum()
                if gross > 0:
                    w = w / gross * gross_leverage

        w.name = d
        weights.append(w)

    W = pd.DataFrame(weights).sort_index()
    W.index.name = "date"
    return W
