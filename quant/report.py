from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_report(
    out_dir: Path,
    equity_curve: pd.Series,
    daily_pnl: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    summary: pd.Series
):
    # Save summary
    summary.to_csv(out_dir / "summary.csv", header=True)

    # Equity curve plot
    plt.figure()
    equity_curve.plot()
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    # Daily returns histogram
    plt.figure()
    daily_pnl.hist(bins=50)
    plt.title("Daily PnL Histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "daily_pnl_hist.png", dpi=150)
    plt.close()

    # Turnover
    plt.figure()
    turnover.rolling(21).mean().plot()
    plt.title("Turnover (21D MA)")
    plt.tight_layout()
    plt.savefig(out_dir / "turnover.png", dpi=150)
    plt.close()

    # Save weights snapshot
    weights.tail(50).to_csv(out_dir / "weights_tail.csv")