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

    # ===== features =====
    # 动量
    mom = px / px.shift(lookback) - 1.0
    # 短期反转
    mr = -(px / px.shift(5) - 1.0)
    # 波动率
    vol = rets.rolling(lookback).std()

    def zscore_cs(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    features = {
        "mom": zscore_cs(mom),
        "mr": zscore_cs(mr),
        "vol": zscore_cs(vol),
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
    y = px.shift(-label_horizon) / px - 1.0  # t+1收益率
    y_long = y.stack().rename("y").reset_index()
    y_long.columns = ["date", "symbol", "y"]

    data = X_long.merge(y_long, on=["date", "symbol"], how="inner")

    # 丢掉前期不完整数据
    data = data.dropna()
    data = data[data["date"] >= data["date"].unique()[min_history]]

    # ===== final output =====
    X = data.set_index(["date", "symbol"])[list(features.keys())]
    y = data.set_index(["date", "symbol"])["y"]

    # 训练/预测用到的日期列表
    meta = {"dates": sorted(X.index.get_level_values(0).unique())}

    return X, y, meta
