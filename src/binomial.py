"""CRR binomial tree pricer — convergence cross-check and American-exercise premium."""

import numpy as np

from src.pricer import black_scholes


def crr_tree_price(
    spot: float, strike: float, rate: float, sigma: float, tmat: float,
    option_type: str = "call", nsteps: int = 100,
) -> dict | None:
    """European option price via a vectorized Cox-Ross-Rubinstein binomial tree."""
    if nsteps < 1:
        return None
    dt = tmat / nsteps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(rate * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)

    j = np.arange(nsteps + 1)
    terminal_spot = spot * (u ** (nsteps - j)) * (d**j)
    if option_type == "call":
        values = np.maximum(terminal_spot - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spot, 0.0)

    for _ in range(nsteps):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])

    return {
        "price": float(values[0]), "spot": spot, "strike": strike, "rate": rate,
        "vol": sigma, "time": tmat, "type": option_type, "nsteps": nsteps,
    }


def crr_convergence(
    spot: float, strike: float, rate: float, sigma: float, tmat: float,
    option_type: str = "call", nsteps_grid: list[int] = None,
) -> dict | None:
    """Show CRR price -> BS price as N -> infinity."""
    if nsteps_grid is None:
        nsteps_grid = [10, 25, 50, 100, 200, 500, 1000]
    bs_price = black_scholes(spot, strike, rate, sigma, tmat, option_type)["price"]
    crr_prices, abs_error = [], []
    for n in nsteps_grid:
        crr_price = crr_tree_price(spot, strike, rate, sigma, tmat, option_type, n)["price"]
        crr_prices.append(crr_price)
        abs_error.append(abs(crr_price - bs_price))
    return {"nsteps_grid": nsteps_grid, "bs_price": bs_price, "crr_prices": crr_prices, "abs_error": abs_error}


def _crr_american_price(spot, strike, rate, sigma, tmat, option_type, nsteps):
    """CRR tree with early exercise checked at every node."""
    dt = tmat / nsteps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(rate * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)

    j = np.arange(nsteps + 1)
    spot_at_step = spot * (u ** (nsteps - j)) * (d**j)
    if option_type == "call":
        values = np.maximum(spot_at_step - strike, 0.0)
    else:
        values = np.maximum(strike - spot_at_step, 0.0)

    for step in range(nsteps - 1, -1, -1):
        j = np.arange(step + 1)
        spot_at_step = spot * (u ** (step - j)) * (d**j)
        continuation = disc * (p * values[:-1] + (1 - p) * values[1:])
        intrinsic = (
            np.maximum(spot_at_step - strike, 0.0)
            if option_type == "call"
            else np.maximum(strike - spot_at_step, 0.0)
        )
        values = np.maximum(continuation, intrinsic)

    return float(values[0])


def check_american_premia(
    spot: float, strike: float, rate: float, sigma: float, tmat: float,
    nsteps: int = 50, dt: list[float] = None,
) -> dict | None:
    """Early-exercise premium (American - European put) across a grid of step sizes.

    Puts only: a non-dividend American call never exercises early, so its
    premium is identically zero and not informative to plot.
    """
    if dt is None:
        dt = [tmat / nsteps, tmat / (nsteps * 2), tmat / (nsteps * 5)]
    premium = []
    for step_size in dt:
        n = max(int(round(tmat / step_size)), 1)
        euro = crr_tree_price(spot, strike, rate, sigma, tmat, "put", n)["price"]
        amer = _crr_american_price(spot, strike, rate, sigma, tmat, "put", n)
        premium.append(amer - euro)
    return {"dt": dt, "premium": premium}
