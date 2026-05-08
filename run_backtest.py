import argparse
from pathlib import Path

from quant.config import BacktestConfig
from quant.data import load_prices_cached_yfinance, load_prices_csv_panel
from quant.features import make_features_and_labels
from quant.model import walkforward_train_predict
from quant.portfolio import build_portfolio_from_scores
from quant.backtest import run_backtest
from quant.metrics import compare_to_benchmarks, summarize_performance
from quant.report import make_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["csv", "yfinance"], default="yfinance")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./out")
    ap.add_argument("--refresh_data", action="store_true")
    args = ap.parse_args()

    cfg = BacktestConfig()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    if args.mode == "csv":
        prices = load_prices_csv_panel(Path(args.data_dir))
    else:
        prices = load_prices_cached_yfinance(
            cfg.universe,
            cfg.start,
            cfg.end,
            data_dir=Path(args.data_dir),
            refresh=args.refresh_data,
        )

    # 2) Features & labels (next-day return)
    X, y, meta = make_features_and_labels(
        prices=prices,
        lookback=cfg.lookback,
        label_horizon=cfg.label_horizon,
        min_history=cfg.min_history,
    )

    # 3) Walk-forward train + predict scores
    scores = walkforward_train_predict(
        X=X,
        y=y,
        meta=meta,
        train_window=cfg.train_window,
        retrain_freq=cfg.retrain_freq,
        model_name=cfg.model_name,
        alpha=cfg.ridge_alpha,
    )

    # 4) Portfolio construction
    weights = build_portfolio_from_scores(
        scores=scores,
        mode=cfg.portfolio_mode,
        long_frac=cfg.long_frac,
        short_frac=cfg.short_frac,
        gross_leverage=cfg.gross_leverage,
        net_exposure=cfg.net_exposure,
        dollar_neutral=cfg.dollar_neutral,
        rebalance_freq=cfg.rebalance_freq,
    )

    # 5) Backtest with costs
    bt = run_backtest(
        prices=prices,
        weights=weights,
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
    )

    # 6) Metrics + report
    summary = summarize_performance(
        bt["equity_curve"], bt["daily_pnl"], turnover=bt["turnover"]
    )
    print(summary.to_string())
    benchmark_summary = compare_to_benchmarks(
        prices=prices,
        strategy_pnl=bt["daily_pnl"],
        strategy_equity=bt["equity_curve"],
    )
    print("\nBenchmark comparison:")
    print(benchmark_summary[["CAGR", "AnnVol", "Sharpe", "MaxDrawdown", "TotalReturn"]])

    make_report(
        out_dir=out_dir,
        equity_curve=bt["equity_curve"],
        daily_pnl=bt["daily_pnl"],
        turnover=bt["turnover"],
        weights=weights,
        summary=summary,
        benchmark_summary=benchmark_summary,
    )


if __name__ == "__main__":
    main()
