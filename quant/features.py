from __future__ import annotations
import pandas as pd


def make_features_and_labels(
    prices: pd.DataFrame,
    lookback: int = 20,
    label_horizon: int = 1,
    min_history: int = 60,
):
    # 对价格矩阵按日期排序并前向填充
    px = prices.sort_index().ffill()
    # 计算收益率
    rets = px.pct_change()

    def zscore_cs(df):
        std = df.std(axis=1).mask(lambda s: s == 0.0)
        return df.sub(df.mean(axis=1), axis=0).div(std, axis=0)

    def trailing_return(window: int):
        return px / px.shift(window) - 1.0

    # ===== features =====
    # Cross-sectional ETF rotation signals. Keep them rank-friendly and avoid
    # absolute market direction so the model learns relative winners.
    mom = trailing_return(lookback)
    mom_60 = trailing_return(60)
    mom_120 = trailing_return(120)
    rev_5 = -trailing_return(5)
    vol = rets.rolling(lookback).std()
    vol_60 = rets.rolling(60).std()
    drawdown_60 = px / px.rolling(60).max() - 1.0

    features = {
        "mom": zscore_cs(mom),
        "mom_60": zscore_cs(mom_60),
        "mom_120": zscore_cs(mom_120),
        "rev_5": zscore_cs(rev_5),
        "vol": zscore_cs(vol),
        "vol_60": zscore_cs(vol_60),
        "drawdown_60": zscore_cs(drawdown_60),
    }

    # 宽表转长表 (date, symbol, feature) → long
    X_list = []
    for name, df in features.items():
        x = df.stack().rename(name).reset_index()
        x.columns = ["date", "symbol", name]
        X_list.append(x)

    X_long = X_list[0]
    for x in X_list[1:]:
        X_long = X_long.merge(x, on=["date", "symbol"])

    # ===== label =====
    y = px.shift(-label_horizon) / px - 1.0
    y = y.sub(y.mean(axis=1), axis=0)  # predict cross-sectional excess return
    y_long = y.stack().rename("y").reset_index()
    y_long.columns = ["date", "symbol", "y"]

    data = X_long.merge(y_long, on=["date", "symbol"], how="inner")

    # 丢掉前期不完整数据
    data = data.dropna()
    if data.empty:
        raise ValueError("No feature/label rows available after cleaning price data.")
    if len(data["date"].unique()) <= min_history:
        raise ValueError(
            f"Need more than {min_history} clean dates, got {len(data['date'].unique())}."
        )
    data = data[data["date"] >= data["date"].unique()[min_history]]

    # ===== final output =====
    X = data.set_index(["date", "symbol"])[list(features.keys())]
    y = data.set_index(["date", "symbol"])["y"]

    # 训练/预测用到的日期列表
    meta = {"dates": sorted(X.index.get_level_values(0).unique())}

    return X, y, meta
