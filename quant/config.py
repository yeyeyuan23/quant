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
    label_horizon: int = 1
    min_history: int = 60

    # Walk-forward
    train_window: int = 252 * 2      # 2y rolling training
    retrain_freq: int = 21           # monthly retrain

    # Model
    model_name: str = "ridge"
    ridge_alpha: float = 10.0

    # Portfolio
    long_frac: float = 0.2
    short_frac: float = 0.2
    gross_leverage: float = 1.0
    dollar_neutral: bool = True

    # Costs
    commission_bps: float = 0.5
    slippage_bps: float = 1.0