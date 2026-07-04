"""CRR binomial tree pricing — cross-check against BS closed form, American exercise support."""


def crr_tree_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
    nsteps: int = 100,
) -> dict | None:
    """Cox-Ross-Rubinstein binomial tree European pricing.

    Args:
        spot (float): S0
        strike (float): K
        rate (float): r
        sigma (float): annualised σ
        tmat (float): T in years
        option_type (str): 'call' or 'put' — also enables American early-exercise flag at end.
        nsteps (int): lattice depth N

    Returns:
        dict with keys: price, spot, strike, rate, vol, time, type, nsteps.
    """


def crr_convergence(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    option_type: str = "call",
    nsteps_grid: list[int] | None = None,
) -> dict | None:
    """Compute BS price and CRR price across increasing N; return convergence table."""


def check_american_premia(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    tmat: float,
    nsteps: int = 50,
    dt: list[float] | None = None,
) -> dict | None:
    """Show early exercise premium decay (European − American) for various Δt."""
