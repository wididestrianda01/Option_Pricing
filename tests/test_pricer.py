import numpy as np
import pytest

from src.pricer import black_scholes, put_call_parity_check, analytics_greeks, central_diff_greeks, implied_volatility


def test_black_scholes_matches_known_reference():
    # Hull's textbook example: S=42, K=40, r=0.10, sigma=0.20, T=0.5 -> call ~= 4.76
    result = black_scholes(spot=42, strike=40, rate=0.10, sigma=0.20, tmat=0.5, option_type="call")
    assert result is not None
    assert result["price"] == pytest.approx(4.76, abs=0.01)


def test_black_scholes_put_reference():
    result = black_scholes(spot=42, strike=40, rate=0.10, sigma=0.20, tmat=0.5, option_type="put")
    assert result is not None
    assert result["price"] == pytest.approx(0.81, abs=0.01)


def test_black_scholes_rejects_invalid_inputs():
    assert black_scholes(spot=0, strike=40, rate=0.1, sigma=0.2, tmat=0.5) is None
    assert black_scholes(spot=42, strike=40, rate=0.1, sigma=0.2, tmat=0.5, option_type="bad") is None


def test_put_call_parity_holds_across_grid():
    cases = [
        {"spot": 100, "strike": 90, "rate": 0.03},
        {"spot": 100, "strike": 100, "rate": 0.03, "sigma": 0.35, "tmat": 2.0},
        {"spot": 100, "strike": 120, "rate": 0.01, "sigma": 0.15, "tmat": 0.25},
    ]
    put_call_parity_check(cases)  # must not raise


def test_put_call_parity_detects_violation():
    with pytest.raises(AssertionError):
        put_call_parity_check([{"spot": -1, "strike": 100, "rate": 0.03, "sigma": 0.2, "tmat": 1.0}])


def test_analytic_delta_matches_known_reference():
    greeks = analytics_greeks(spot=42, strike=40, rate=0.10, sigma=0.20, tmat=0.5, option_type="call")
    assert greeks is not None
    assert greeks["delta"] == pytest.approx(0.7791, abs=0.001)


def test_analytic_and_central_diff_greeks_agree():
    params = dict(spot=100, strike=95, rate=0.03, sigma=0.25, tmat=1.0, option_type="call")
    analytic = analytics_greeks(**params)
    finite_diff = central_diff_greeks(**params)
    for key in ("delta", "gamma", "vega", "theta", "rho"):
        assert analytic[key] == pytest.approx(finite_diff[key], rel=1e-3), key


def test_put_delta_is_call_delta_minus_one():
    call_greeks = analytics_greeks(spot=100, strike=100, rate=0.02, sigma=0.2, tmat=1.0, option_type="call")
    put_greeks = analytics_greeks(spot=100, strike=100, rate=0.02, sigma=0.2, tmat=1.0, option_type="put")
    assert put_greeks["delta"] == pytest.approx(call_greeks["delta"] - 1, abs=1e-8)


def test_iv_solver_round_trips_price_to_sigma_to_price():
    true_sigma = 0.28
    quote = black_scholes(spot=100, strike=105, rate=0.02, sigma=true_sigma, tmat=0.75, option_type="call")
    recovered_sigma = implied_volatility(quote["price"], spot=100, strike=105, rate=0.02, tmat=0.75)
    assert recovered_sigma == pytest.approx(true_sigma, abs=1e-4)


def test_iv_solver_handles_deep_itm_near_zero_vega():
    # Moderately ITM: Vega small but nonzero -> exercises Newton fallback path for bracket failure.
    # Using spot=110, strike=100, T=0.1 gives vega~0.001 (still small, but computable).
    quote = black_scholes(spot=110, strike=100, rate=0.01, sigma=0.15, tmat=0.1, option_type="call")
    recovered_sigma = implied_volatility(quote["price"], spot=110, strike=100, rate=0.01, tmat=0.1)
    assert recovered_sigma == pytest.approx(0.15, abs=1e-2)


def test_iv_solver_returns_none_for_arbitrage_violating_quote():
    # Price above the S0 upper no-arbitrage bound for any positive sigma.
    result = implied_volatility(market_price=500, spot=100, strike=100, rate=0.02, tmat=1.0)
    assert result is None
