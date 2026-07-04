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
    path = generate_gbm_path(spot_start=100, sigma=0.2, rate=0.03, tmat=1.0, nsteps=252, seed=42)
    errors = {
        freq: abs(delta_rebalance(path, "call", 100, freq)["hedge_error"])
        for freq in ("monthly", "weekly", "daily")
    }
    assert errors["daily"] <= errors["weekly"] <= errors["monthly"]


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
