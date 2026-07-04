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


def _freq_to_steps(freq: str | int) -> int:
    """daily=1, weekly=5, monthly=21 trading-day steps; int freq bypasses presets."""
    if isinstance(freq, int):
        return freq
    if freq not in _FREQ_STEPS:
        raise ValueError(f"Unknown frequency preset: {freq!r}")
    return _FREQ_STEPS[freq]


def delta_rebalance(path_data: dict, option_type: str, strike: float, freq: str | int) -> dict:
    """Discretely rebalanced replicating portfolio; returns hedging-error P&L vs terminal payoff."""
    times, spot_path = path_data["times"], path_data["spot"]
    rate, sigma = path_data["rate"], path_data["sigma"]
    tmat = times[-1]
    step = _freq_to_steps(freq)

    rebalance_idx = list(range(0, len(times) - 1, step))
    if rebalance_idx[-1] != len(times) - 2:
        rebalance_idx.append(len(times) - 2)

    cash = black_scholes(spot_path[0], strike, rate, sigma, tmat, option_type)["price"]
    shares = 0.0
    prev_idx = 0

    for idx in rebalance_idx:
        t_remaining = tmat - times[idx]
        if t_remaining <= 0:
            break
        new_shares = analytics_greeks(spot_path[idx], strike, rate, sigma, t_remaining, option_type)["delta"]
        dt = times[idx] - times[prev_idx]
        cash *= np.exp(rate * dt)
        cash -= (new_shares - shares) * spot_path[idx]
        shares = new_shares
        prev_idx = idx

    cash *= np.exp(rate * (times[-1] - times[prev_idx]))
    portfolio_value = cash + shares * spot_path[-1]
    payoff = (
        max(spot_path[-1] - strike, 0.0) if option_type == "call" else max(strike - spot_path[-1], 0.0)
    )

    return {"portfolio_value": float(portfolio_value), "payoff": float(payoff),
            "hedge_error": float(portfolio_value - payoff)}


def hedge_pnl_analysis(
    spot_path: np.ndarray, sigma: float, rate: float, spot_price: float,
    strikes: list[float] = None, frequencies: list[str] = None,
) -> dict:
    """Run the full delta-hedge simulation loop across a price path and report hedging-error P&L.

    `spot_path` is assumed sampled at the daily (252/year trading-day) convention.
    """
    if strikes is None:
        strikes = [0.9 * spot_price, spot_price, 1.1 * spot_price]
    if frequencies is None:
        frequencies = ["daily", "weekly", "monthly"]

    nsteps = len(spot_path) - 1
    tmat = nsteps / 252
    times = np.linspace(0, tmat, nsteps + 1)
    path_data = {"times": times, "spot": np.asarray(spot_path), "rate": rate, "sigma": sigma}

    results = []
    for strike in strikes:
        for freq in frequencies:
            hedge = delta_rebalance(path_data, "call", strike, freq)
            results.append({"strike": strike, "frequency": freq, "hedge_error": hedge["hedge_error"]})

    return {"results": results, "strikes": strikes, "frequencies": frequencies}


def hedge_error_vs_frequency_table(analysis_result: dict) -> pd.DataFrame:
    """Pivot hedge_pnl_analysis results into a strike x frequency abs-hedge-error table."""
    df = pd.DataFrame(analysis_result["results"])
    df["abs_hedge_error"] = df["hedge_error"].abs()
    return df.pivot(index="strike", columns="frequency", values="abs_hedge_error")


def compare_continuous_vs_discrete_hedge(
    path_data: dict, option_type: str, strike: float, rate: float, sigma: float
) -> dict:
    """Compare theoretical continuous-hedge error (~0 by the replication theorem)
    against the realised discrete (daily) hedge error."""
    assert np.isclose(rate, path_data["rate"]) and np.isclose(sigma, path_data["sigma"]), (
        "rate/sigma must match the path's own parameters"
    )
    discrete = delta_rebalance(path_data, option_type, strike, freq=1)
    return {
        "continuous_hedge_error": 0.0,
        "discrete_hedge_error": discrete["hedge_error"],
        "discretisation_gap": discrete["hedge_error"],
    }
