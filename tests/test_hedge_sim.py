import numpy as np
import pytest

from src.hedge_sim import generate_gbm_path


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
