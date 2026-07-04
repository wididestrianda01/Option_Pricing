"""Black-Scholes European options pricer, Greeks (analytical + finite-difference), and IV solver."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


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
    bs = black_scholes(spot, strike, rate, sigma, tmat, option_type)
    if bs is None:
        return None
    d1, d2 = bs["d1"], bs["d2"]
    sqrt_t = np.sqrt(tmat)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-rate * tmat)

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = -(spot * pdf_d1 * sigma) / (2 * sqrt_t) - rate * strike * disc * norm.cdf(d2)
        rho = strike * tmat * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(spot * pdf_d1 * sigma) / (2 * sqrt_t) + rate * strike * disc * norm.cdf(-d2)
        rho = -strike * tmat * disc * norm.cdf(-d2)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


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
    def price_at(s=spot, sig=sigma, t=tmat, r=rate):
        bs = black_scholes(s, strike, r, sig, t, option_type)
        return bs["price"] if bs else None

    h_s, h_sig, h_t, h_r = spot * vega_eps, vega_eps, vega_eps, vega_eps

    p_up_s, p_dn_s = price_at(s=spot + h_s), price_at(s=spot - h_s)
    p0 = price_at()
    p_up_sig, p_dn_sig = price_at(sig=sigma + h_sig), price_at(sig=sigma - h_sig)
    p_up_t, p_dn_t = price_at(t=tmat + h_t), price_at(t=max(tmat - h_t, 1e-8))
    p_up_r, p_dn_r = price_at(r=rate + h_r), price_at(r=rate - h_r)

    prices = (p_up_s, p_dn_s, p0, p_up_sig, p_dn_sig, p_up_t, p_dn_t, p_up_r, p_dn_r)
    if any(p is None for p in prices):
        return None

    return {
        "delta": (p_up_s - p_dn_s) / (2 * h_s),
        "gamma": (p_up_s - 2 * p0 + p_dn_s) / (h_s**2),
        "vega": (p_up_sig - p_dn_sig) / (2 * h_sig),
        "theta": (p_dn_t - p_up_t) / (2 * h_t),
        "rho": (p_up_r - p_dn_r) / (2 * h_r),
    }


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
    """Solve C_BS(σ) = market_price via Brent's method (call options),
    with a Newton + Vega-guard fallback for cases Brent cannot bracket."""
    vol_lo, vol_hi, tol, max_iter = 1e-6, 5.0, 1e-8, 100

    def f(sigma):
        return black_scholes(spot, strike, rate, sigma, tmat, "call")["price"] - market_price

    try:
        sigma_brent = float(brentq(f, vol_lo, vol_hi, xtol=tol, maxiter=max_iter))
        # Validate that the solution is not at a boundary (which indicates a flat region)
        greeks = analytics_greeks(spot, strike, rate, sigma_brent, tmat, "call")
        if greeks is not None and abs(greeks["vega"]) > 1e-8:
            return sigma_brent
        # If vega is too small, the IV is unreliable; fall through to Newton
    except ValueError:
        pass

    # Newton's method with vega-guard fallback
    sigma = 0.2
    for _ in range(max_iter):
        bs = black_scholes(spot, strike, rate, sigma, tmat, "call")
        greeks = analytics_greeks(spot, strike, rate, sigma, tmat, "call")
        if bs is None or greeks is None or abs(greeks["vega"]) < 1e-10:
            return None
        diff = bs["price"] - market_price
        if abs(diff) < tol:
            return float(sigma)
        sigma -= diff / greeks["vega"]
        if sigma <= 0:
            return None
    return None
