from __future__ import annotations
from pathlib import Path
import pandas as pd


def _required_symbols_present(data_dir: Path, universe) -> bool:
    return all((data_dir / f"{sym}.csv").exists() for sym in universe)


def _covers_date_range(prices: pd.DataFrame, start: str, end: str) -> bool:
    if prices.empty:
        return False
    start_ts = pd.Timestamp(start) + pd.Timedelta(days=7)
    # yfinance treats end as exclusive; allow a short weekend/holiday gap.
    end_ts = pd.Timestamp(end) - pd.Timedelta(days=7)
    return prices.index.min() <= start_ts and prices.index.max() >= end_ts


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


def save_prices_csv_panel(prices: pd.DataFrame, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for sym in prices.columns:
        s = prices[sym].dropna().rename("Close")
        s.to_csv(data_dir / f"{sym}.csv", index_label="Date")


def load_prices_cached_yfinance(
    universe,
    start: str,
    end: str,
    data_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Use local per-symbol CSV files when they cover the requested universe/date
    range. Download from yfinance only when the cache is missing or stale.
    """
    universe = [str(sym).upper() for sym in universe]
    if not refresh and _required_symbols_present(data_dir, universe):
        prices = load_prices_csv_panel(data_dir).reindex(columns=universe)
        if _covers_date_range(prices, start, end):
            print(f"Loaded cached prices from {data_dir}")
            return prices.loc[pd.Timestamp(start) :]

    prices = load_prices_yfinance(universe, start, end)
    save_prices_csv_panel(prices, data_dir)
    print(f"Downloaded prices from yfinance and cached them in {data_dir}")
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
    if px.empty or px.dropna(how="all").empty:
        raise ValueError("No price data loaded from yfinance.")
    return px
