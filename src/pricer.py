"""Black-Scholes European options pricer, Greeks (analytical + finite-difference), and IV solver."""


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
) -> dict | None:
    """European Black-Scholes closed-form price.

    Args:
        spot (float): Underlying spot price S0.
        strike (float): Option strike K.
        rate (float): Continuous risk-free rate r.
        sigma (float): Annualised volatility σ.
        tmat (float): Time to maturity in years T.
        option_type (str): "call" or "put".

    Returns:
        dict with keys: spot, strike, rate, vol, time, price, type, d1, d2  OR None on failure.
    """


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
    """Validate C - P = S - K·exp(-rT) across a grid of inputs.

    parities – list of dicts with spot, strike, rate (used to recompute BS prices)."""


def implied_volatility(
    market_price: float, spot: float, strike: float, rate: float, tmat: float
) -> float | None:
    """Solve C_BS(σ) = market_price for σ via Brent's method."""
