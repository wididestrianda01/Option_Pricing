"""Synthetic OMXS30 volatility surface construction and IV-inversion self-recovery test."""

import numpy as np

from src.pricer import black_scholes, implied_volatility

_TERM_SLOPE = 0.03  # mild upward vol term structure (documents the spec's "+ term structure" clause)


def build_synthetic_smile(
    spot: float, rate: float, sigma_base: float, skew_slope: float = -0.2,
    strikes_grid: np.ndarray = None, maturities: list[float] = None,
) -> dict | None:
    """Construct a synthetic call-price surface under an assumed vol surface
    (base + skew*moneyness + term structure). Recovering this via IV inversion is the validation test."""
    if strikes_grid is None:
        strikes_grid = np.linspace(0.8 * spot, 1.2 * spot, 9)
    if maturities is None:
        maturities = [0.25, 0.5, 1.0]

    sigma_true = np.zeros((len(maturities), len(strikes_grid)))
    price = np.zeros_like(sigma_true)

    for i, tmat in enumerate(maturities):
        for k, strike in enumerate(strikes_grid):
            moneyness = np.log(strike / spot)
            sigma = max(sigma_base + skew_slope * moneyness + _TERM_SLOPE * np.sqrt(tmat), 0.01)
            sigma_true[i, k] = sigma
            price[i, k] = black_scholes(spot, strike, rate, sigma, tmat, "call")["price"]

    return {
        "spot": spot, "rate": rate, "strikes": strikes_grid, "maturities": maturities,
        "sigma_true": sigma_true, "price": price,
    }


def invert_iv_surface(synthetic_smile_dict: dict) -> dict | None:
    """Recover sigma_true from price via the IV solver; report recovery error."""
    spot, rate = synthetic_smile_dict["spot"], synthetic_smile_dict["rate"]
    strikes, maturities = synthetic_smile_dict["strikes"], synthetic_smile_dict["maturities"]
    price, sigma_true = synthetic_smile_dict["price"], synthetic_smile_dict["sigma_true"]

    sigma_hat = np.zeros_like(sigma_true)
    for i, tmat in enumerate(maturities):
        for k, strike in enumerate(strikes):
            sigma_hat[i, k] = implied_volatility(price[i, k], spot, strike, rate, tmat)

    abs_error = np.abs(sigma_hat - sigma_true)
    return {
        "sigma_hat": sigma_hat, "sigma_true": sigma_true, "abs_error": abs_error,
        "max_abs_error": float(np.max(abs_error)), "mean_abs_error": float(np.mean(abs_error)),
    }


def plot_smile_and_surface(synthetic_smile_dict): ...


def surface_skew_analysis(synthetic_smile_dict) -> dict | None:
    """Compute ATM, skew, and term-structure metrics; flag skew magnitude relative to BS assumption."""
