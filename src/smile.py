"""Synthetic OMXS30 volatility surface construction and IV-inversion self-recovery test."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

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
    """Recover sigma_true from price via the IV solver; report recovery error.

    Raises ValueError if required keys are missing from synthetic_smile_dict.
    IV solver failures (None returns) are stored as np.nan; error metrics use NaN-aware reductions.
    """
    # Validate required keys
    required_keys = {"spot", "rate", "strikes", "maturities", "price", "sigma_true"}
    missing_keys = required_keys - set(synthetic_smile_dict.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in synthetic_smile_dict: {sorted(missing_keys)}")

    spot, rate = synthetic_smile_dict["spot"], synthetic_smile_dict["rate"]
    strikes, maturities = synthetic_smile_dict["strikes"], synthetic_smile_dict["maturities"]
    price, sigma_true = synthetic_smile_dict["price"], synthetic_smile_dict["sigma_true"]

    sigma_hat = np.zeros_like(sigma_true)
    for i, tmat in enumerate(maturities):
        for k, strike in enumerate(strikes):
            iv = implied_volatility(price[i, k], spot, strike, rate, tmat)
            # Store np.nan if IV solver fails (returns None), otherwise store the IV
            sigma_hat[i, k] = np.nan if iv is None else iv

    abs_error = np.abs(sigma_hat - sigma_true)
    return {
        "sigma_hat": sigma_hat, "sigma_true": sigma_true, "abs_error": abs_error,
        "max_abs_error": float(np.nanmax(abs_error)), "mean_abs_error": float(np.nanmean(abs_error)),
    }


def surface_skew_analysis(synthetic_smile_dict: dict) -> dict | None:
    """ATM vol per maturity, low-strike/high-strike skew, and term-structure slope.

    Raises ValueError if required keys are missing from synthetic_smile_dict.
    """
    # Validate required keys
    required_keys = {"spot", "strikes", "maturities", "sigma_true"}
    missing_keys = required_keys - set(synthetic_smile_dict.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in synthetic_smile_dict: {sorted(missing_keys)}")

    spot = synthetic_smile_dict["spot"]
    strikes = np.asarray(synthetic_smile_dict["strikes"])
    maturities = synthetic_smile_dict["maturities"]
    sigma_true = synthetic_smile_dict["sigma_true"]

    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    atm_vol = sigma_true[:, atm_idx].tolist()
    skew = (sigma_true[:, 0] - sigma_true[:, -1]).tolist()
    term_structure_slope = float(np.polyfit(maturities, atm_vol, 1)[0])

    return {
        "atm_vol": atm_vol, "skew": skew, "term_structure_slope": term_structure_slope,
        "flat_vol_violated": bool(max(abs(s) for s in skew) > 0.01),
    }


def plot_smile_and_surface(synthetic_smile_dict: dict) -> plt.Figure:
    """2D smile slices (per maturity) + 3D IV surface.

    Raises ValueError if required keys are missing from synthetic_smile_dict.
    """
    # Validate required keys
    required_keys = {"strikes", "maturities", "sigma_true"}
    missing_keys = required_keys - set(synthetic_smile_dict.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in synthetic_smile_dict: {sorted(missing_keys)}")

    strikes = synthetic_smile_dict["strikes"]
    maturities = synthetic_smile_dict["maturities"]
    sigma_true = synthetic_smile_dict["sigma_true"]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    for i, tmat in enumerate(maturities):
        ax1.plot(strikes, sigma_true[i], label=f"T={tmat}y")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Implied vol")
    ax1.set_title("Volatility smile")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    ax2.plot_surface(strike_grid, maturity_grid, sigma_true, cmap="viridis")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("Implied vol")
    ax2.set_title("Volatility surface")

    return fig
