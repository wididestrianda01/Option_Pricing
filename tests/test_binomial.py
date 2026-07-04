import numpy as np
import pytest
import QuantLib as ql

from src.pricer import black_scholes
from src.binomial import crr_tree_price, crr_convergence


def _quantlib_bs_price(spot, strike, rate, sigma, tmat, option_type="call"):
    calc_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calc_date
    maturity_date = calc_date + int(round(tmat * 365))

    ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_curve = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, rate, ql.Actual365Fixed()))
    vol_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calc_date, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )
    process = ql.BlackScholesProcess(spot_handle, rate_curve, vol_curve)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()


def test_crr_matches_bs_at_large_nsteps():
    params = dict(spot=100, strike=100, rate=0.03, sigma=0.2, tmat=1.0, option_type="call")
    bs_price = black_scholes(**params)["price"]
    crr_price = crr_tree_price(**params, nsteps=500)["price"]
    assert crr_price == pytest.approx(bs_price, abs=0.05)


def test_crr_convergence_error_shrinks_monotonically():
    result = crr_convergence(spot=100, strike=100, rate=0.03, sigma=0.2, tmat=1.0, option_type="call")
    errors = result["abs_error"]
    # Binomial convergence oscillates step-to-step (odd/even N parity); check the
    # smoothed trend (every-other-step) is monotonically non-increasing.
    even_errors = errors[::2]
    assert all(even_errors[i] >= even_errors[i + 1] - 1e-9 for i in range(len(even_errors) - 1))


def test_crr_cross_checks_against_quantlib():
    params = dict(spot=100, strike=95, rate=0.02, sigma=0.25, tmat=0.5, option_type="call")
    our_price = crr_tree_price(**params, nsteps=1000)["price"]
    ql_price = _quantlib_bs_price(**params)
    assert our_price == pytest.approx(ql_price, abs=0.05)
