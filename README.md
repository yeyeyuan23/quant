# Quant Factor + ML Alpha (Walk-forward) with Cost-aware Backtest

A production-style, end-to-end quantitative equity alpha pipeline:
- Data ingestion (CSV / optional yfinance)
- Feature engineering: momentum, mean reversion, volatility, volume signals
- Walk-forward training (rolling window) with Ridge regression
- Cross-sectional portfolio construction (rank-based weights, neutrality optional)
- Cost-aware backtest (commission + slippage, turnover tracked)
- Full performance & risk metrics + plots export

## Run
pip install -r requirements.txt
python run_backtest.py --mode csv --data_dir ./data --out_dir ./out
# Or (if internet is available):
python run_backtest.py --mode yfinance --out_dir ./out