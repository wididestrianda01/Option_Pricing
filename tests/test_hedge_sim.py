import numpy as np
import pytest

from src.hedge_sim import (
    generate_gbm_path,
    delta_rebalance,
    hedge_pnl_analysis,
    hedge_error_vs_frequency_table,
    compare_continuous_vs_discrete_hedge,
)


def test_gbm_path_reproducible_under_fixed_seed():
    path_a = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=42)
    path_b = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=42)
    assert np.array_equal(path_a["spot"], path_b["spot"])


def test_gbm_path_differs_across_seeds():
    path_a = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=1)
    path_b = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=2)
    assert not np.array_equal(path_a["spot"], path_b["spot"])


def test_gbm_path_shape_and_start_value():
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, nsteps=252, seed=42)
    assert len(path["spot"]) == 253
    assert path["spot"][0] == pytest.approx(100)
    assert path["dt"] == pytest.approx(1.0 / 252)


def test_delta_rebalance_returns_hedge_error():
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=42)
    result = delta_rebalance(path, option_type="call", strike=100, freq="daily")
    assert set(result) == {"portfolio_value", "payoff", "hedge_error"}
    assert isinstance(result["hedge_error"], float)


def test_hedge_error_shrinks_monotonically_with_frequency():
    # Test that mean abs hedge error converges monotonically with rebalance frequency
    # across multiple sample paths, not on a single fixed path (path-level monotonicity
    # is path-dependent; convergence is guaranteed only in expectation across many paths).
    num_seeds = 40

    errors_by_freq = {"daily": [], "weekly": [], "monthly": []}

    for seed in range(num_seeds):
        path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, nsteps=252, seed=seed)
        for freq in ("daily", "weekly", "monthly"):
            hedge_error = delta_rebalance(path, "call", 100, freq)["hedge_error"]
            errors_by_freq[freq].append(abs(hedge_error))

    mean_errors = {freq: np.mean(errors) for freq, errors in errors_by_freq.items()}

    # Monotonic convergence: more frequent rebalancing → smaller mean hedge error
    # Allow small numerical tolerance for near-ties due to RNG variance
    tolerance = 1e-10
    assert mean_errors["daily"] <= mean_errors["weekly"] + tolerance, (
        f"Expected daily mean ≤ weekly mean; got daily={mean_errors['daily']:.6f}, "
        f"weekly={mean_errors['weekly']:.6f}"
    )
    assert mean_errors["weekly"] <= mean_errors["monthly"] + tolerance, (
        f"Expected weekly mean ≤ monthly mean; got weekly={mean_errors['weekly']:.6f}, "
        f"monthly={mean_errors['monthly']:.6f}"
    )


def test_int_freq_bypasses_named_presets():
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, nsteps=252, seed=42)
    result_int = delta_rebalance(path, "call", 100, freq=1)
    result_daily = delta_rebalance(path, "call", 100, freq="daily")
    assert result_int["hedge_error"] == pytest.approx(result_daily["hedge_error"])


def test_hedge_pnl_analysis_and_table():
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, nsteps=252, seed=42)
    analysis = hedge_pnl_analysis(path["spot"], sigma=0.2, rate=0.03, spot_price=100)
    table = hedge_error_vs_frequency_table(analysis)
    assert table.shape == (3, 3)  # 3 strikes x 3 frequencies


def test_compare_continuous_vs_discrete_hedge():
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, seed=42)
    result = compare_continuous_vs_discrete_hedge(path, "call", 100, rate=0.03, sigma=0.2)
    assert result["continuous_hedge_error"] == 0.0
    assert result["discretisation_gap"] == pytest.approx(result["discrete_hedge_error"])
