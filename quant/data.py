from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_prices_csv_panel(data_dir: Path) -> pd.DataFrame:
    """
    Expect files like:
      data_dir/SPY.csv, QQQ.csv ...
    Each csv must have columns: Date, Close (optionally Adj Close)
    Returns: prices DataFrame [date x symbol] float
    """
    frames = []
    for fp in sorted(data_dir.glob("*.csv")):
        sym = fp.stem.upper()
        df = pd.read_csv(fp)
        if "Date" not in df.columns:
            raise ValueError(f"{fp} missing Date column")
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if col not in df.columns:
            raise ValueError(f"{fp} missing Close/Adj Close column")
        s = pd.to_datetime(df["Date"])
        px = pd.Series(df[col].values, index=s, name=sym).sort_index()
        frames.append(px)
    if not frames:
        raise ValueError(f"No csv files found in {data_dir}")
    prices = pd.concat(frames, axis=1).sort_index().ffill()
    return prices


def load_prices_yfinance(universe, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        list(universe), start=start, end=end, auto_adjust=True, progress=False
    )
    # yfinance returns columns like ('Close', 'SPY') etc
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            # auto_adjust True usually returns 'Close'
            px = df.xs(df.columns.levels[0][0], axis=1, level=0)
    else:
        # single symbol case
        px = df[["Close"]].rename(columns={"Close": universe[0]})
    px = px.sort_index().ffill()
    return px
