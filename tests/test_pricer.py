import numpy as np
import pytest

from src.pricer import black_scholes, put_call_parity_check


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
