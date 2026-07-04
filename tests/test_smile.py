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


def test_invert_iv_surface_handles_none_from_implied_vol(monkeypatch):
    """IV solver failure (returning None) should be stored as np.nan and not crash surface inversion."""
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2)

    # Monkeypatch implied_volatility to return None for the first call
    from src import smile as smile_module
    call_count = [0]
    original_iv = smile_module.implied_volatility

    def mock_iv(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:  # Return None on first call
            return None
        return original_iv(*args, **kwargs)

    monkeypatch.setattr(smile_module, "implied_volatility", mock_iv)

    # Should not raise; should return dict with NaN at the affected cell
    result = invert_iv_surface(smile)
    assert isinstance(result, dict)
    assert np.isnan(result["sigma_hat"][0, 0])  # First cell should be NaN
    # NaN-aware reductions should not crash
    assert np.isnan(result["max_abs_error"]) or isinstance(result["max_abs_error"], float)
    assert np.isnan(result["mean_abs_error"]) or isinstance(result["mean_abs_error"], float)


def test_invert_iv_surface_validates_required_keys():
    """invert_iv_surface should raise ValueError if required keys are missing."""
    incomplete_smile = {
        "spot": 100,
        "rate": 0.02,
        # Missing: strikes, maturities, price, sigma_true
    }

    with pytest.raises(ValueError) as exc_info:
        invert_iv_surface(incomplete_smile)

    # Error message should name the missing keys
    assert "strikes" in str(exc_info.value) or "missing" in str(exc_info.value).lower()


def test_surface_skew_analysis_flags_flat_vol_violation():
    from src.smile import surface_skew_analysis
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2, skew_slope=-0.3)
    result = surface_skew_analysis(smile)
    assert result["flat_vol_violated"] is True
    assert len(result["atm_vol"]) == 3
    assert len(result["skew"]) == 3


def test_plot_smile_and_surface_returns_figure():
    from src.smile import plot_smile_and_surface
    smile = build_synthetic_smile(spot=100, rate=0.02, sigma_base=0.2)
    fig = plot_smile_and_surface(smile)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
