"""Discrete delta-hedging backtest simulation — replicating portfolio P&L analysis."""


def generate_gbm_path(
    spot_start: float,
    sigma: float,
    rate: float,
    tmat: float,
    nsteps: int = 252 * 365,
    seed=42,
    drift="r",
) -> None: ...


def delta_rebalance(
    path_data: dict, option_type: str, strike: float, freq: str | int
) -> None: ...


def hedge_pnl_analysis(
    spot_path: np.ndarray,
    sigma: float,
    rate: float,
    spot_price: float,
    strikes: list[float] = None,
    frequencies: list[str] = None,
) -> dict | None:
    """Run the full delta-hedge simulation loop across a price path and report hedging-error P&L.

    Frequencies tested — e.g. 'daily', 'weekly', 'monthly'. Returns error-by-frequency table."""


def hedge_error_vs_frequency_table(analysis_result: dict) -> None: ...


def compare_continuous_vs_discrete_hedge(
    path_data, option_type, strike, rate, sigma
) -> None: ...
