"""Discrete delta-hedging backtest simulation — replicating-portfolio P&L analysis."""

import numpy as np
import pandas as pd

from src.pricer import black_scholes, analytics_greeks

_FREQ_STEPS = {"daily": 1, "weekly": 5, "monthly": 21}


def generate_gbm_path(
    spot_start: float, sigma: float, rate: float, tmat: float,
    nsteps: int = 252, seed: int = 42, drift: str | float = "r",
) -> dict:
    """Simulate one GBM spot path with a fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    dt = tmat / nsteps
    mu = rate if drift == "r" else drift

    z = rng.standard_normal(nsteps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    spot_path = spot_start * np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    times = np.linspace(0, tmat, nsteps + 1)

    return {"times": times, "spot": spot_path, "dt": dt, "nsteps": nsteps, "sigma": sigma, "rate": rate}


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
