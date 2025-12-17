import argparse
from pathlib import Path

from quant.config import BacktestConfig
from quant.data import load_prices_csv_panel, load_prices_yfinance
from quant.features import make_features_and_labels
from quant.model import walkforward_train_predict
from quant.portfolio import build_portfolio_from_scores
from quant.backtest import run_backtest
from quant.metrics import summarize_performance
from quant.report import make_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["csv", "yfinance"], default="csv")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./out")
    args = ap.parse_args()

    cfg = BacktestConfig()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    if args.mode == "csv":
        prices = load_prices_csv_panel(Path(args.data_dir))
    else:
        prices = load_prices_yfinance(cfg.universe, cfg.start, cfg.end)

    # 2) Features & labels (next-day return)
    X, y, meta = make_features_and_labels(
        prices=prices,
        lookback=cfg.lookback,
        label_horizon=cfg.label_horizon,
        min_history=cfg.min_history
    )

    # 3) Walk-forward train + predict scores
    scores = walkforward_train_predict(
        X=X, y=y, meta=meta,
        train_window=cfg.train_window,
        retrain_freq=cfg.retrain_freq,
        model_name=cfg.model_name,
        alpha=cfg.ridge_alpha
    )

    # 4) Portfolio construction
    weights = build_portfolio_from_scores(
        scores=scores,
        long_frac=cfg.long_frac,
        short_frac=cfg.short_frac,
        gross_leverage=cfg.gross_leverage,
        dollar_neutral=cfg.dollar_neutral
    )

    # 5) Backtest with costs
    bt = run_backtest(
        prices=prices,
        weights=weights,
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps
    )

    # 6) Metrics + report
    summary = summarize_performance(bt["equity_curve"], bt["daily_pnl"])
    print(summary.to_string())

    make_report(
        out_dir=out_dir,
        equity_curve=bt["equity_curve"],
        daily_pnl=bt["daily_pnl"],
        turnover=bt["turnover"],
        weights=weights,
        summary=summary
    )

if __name__ == "__main__":
    main()