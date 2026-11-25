"""Spectrum normalisation helpers of bls_cpu."""
import numpy as np
import pytest

from pytfit5 import bls_cpu


def _naive_rolling_nanmedian(x, width):
    n = x.size
    width = min(width, n)
    out = np.empty(n)
    for i in range(n):
        start = max(0, min(i - width // 2, n - width))
        w = x[start:start + width]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if w.size else np.nan
    return out


@pytest.mark.parametrize("n, width, frac_nan", [(500, 31, 0.0), (500, 31, 0.3), (300, 101, 0.9),
                                                (50, 81, 0.2), (400, 4, 0.1), (5000, 777, 0.25)])
def test_rolling_nanmedian_matches_naive(n, width, frac_nan):
    rng = np.random.default_rng(n + width)
    x = rng.standard_normal(n)
    x[rng.random(n) < frac_nan] = np.nan
    x[::17] = 1.0                       # ties
    np.testing.assert_array_equal(bls_cpu._rolling_nanmedian(x, width), _naive_rolling_nanmedian(x, width))
