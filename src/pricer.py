"""Black-Scholes European options pricer, Greeks (analytical + finite-difference), and IV solver."""

import numpy as np
from scipy.stats import norm


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
) -> dict | None:
    """European Black-Scholes closed-form price (no dividends)."""
    if spot <= 0 or strike <= 0 or sigma <= 0 or tmat <= 0:
        return None
    if option_type not in ("call", "put"):
        return None

    sqrt_t = np.sqrt(tmat)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tmat) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * tmat) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * tmat) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    return {
        "spot": spot, "strike": strike, "rate": rate, "vol": sigma, "time": tmat,
        "price": float(price), "type": option_type, "d1": float(d1), "d2": float(d2),
    }


def analytics_greeks(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
) -> dict | None:
    """Closed-form Greeks (δ, Γ, ν, θ, ρ) for European BS."""


def central_diff_greeks(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
    vega_eps: float = 1e-4,
) -> dict | None:
    """Approximate Greeks via central finite differences for cross-checking analytic formulas."""


def put_call_parity_check(parities: list[dict]) -> None:
    """Assert C - P = S - K*exp(-rT) across a grid of {spot, strike, rate, sigma?, tmat?}."""
    for case in parities:
        spot, strike, rate = case["spot"], case["strike"], case["rate"]
        sigma, tmat = case.get("sigma", 0.2), case.get("tmat", 1.0)
        call = black_scholes(spot, strike, rate, sigma, tmat, "call")
        put = black_scholes(spot, strike, rate, sigma, tmat, "put")
        assert call is not None and put is not None, f"Invalid inputs: {case}"
        lhs = call["price"] - put["price"]
        rhs = spot - strike * np.exp(-rate * tmat)
        assert np.isclose(lhs, rhs, atol=1e-8), f"Parity violated for {case}: {lhs} != {rhs}"


def implied_volatility(
    market_price: float, spot: float, strike: float, rate: float, tmat: float
) -> float | None:
    """Solve C_BS(σ) = market_price for σ via Brent's method."""
