"""Tests for synthetic volatility surface generation and IV recovery."""

import numpy as np
import pytest

from src.smile import build_synthetic_smile, invert_iv_surface


def test_build_synthetic_smile_shapes():
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2)
    assert smile["sigma_true"].shape == (3, 9)
    assert smile["price"].shape == (3, 9)
    assert np.all(smile["sigma_true"] > 0)


def test_invert_iv_surface_self_recovery_below_threshold():
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2)
    result = invert_iv_surface(smile)
    assert result["max_abs_error"] < 1e-4


def test_invert_iv_surface_recovers_skew_direction():
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2, skew_slope=-0.3)
    result = invert_iv_surface(smile)
    # Negative skew_slope -> vol falls as strike rises -> first (lowest-strike) column
    # has higher recovered IV than the last (highest-strike) column, for every maturity.
    assert np.all(result["sigma_hat"][:, 0] > result["sigma_hat"][:, -1])
