from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BacktestConfig:
    # Universe
    universe: List[str] = field(
        default_factory=lambda: "SPY QQQ IWM DIA XLF XLK XLE XLV XLI XLY XLP XLU".split()
    )
    start: str = "2012-01-01"
    end: str = "2024-12-31"

    # Feature/label
    lookback: int = 20
    label_horizon: int = 5
    min_history: int = 126

    # Walk-forward
    train_window: int = 252  # 1y rolling training
    retrain_freq: int = 21  # monthly retrain

    # Model
    model_name: str = "ridge"
    ridge_alpha: float = 10.0

    # Portfolio
    portfolio_mode: str = "long_short"
    long_frac: float = 0.50
    short_frac: float = 0.17
    gross_leverage: float = 1.0
    net_exposure: float = 0.80
    dollar_neutral: bool = False
    rebalance_freq: int = 21  # trading days between portfolio rebalances

    # Costs
    commission_bps: float = 0.5
    slippage_bps: float = 1.0
