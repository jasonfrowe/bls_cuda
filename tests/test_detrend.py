"""The compiled detrender and outlier cut must match the original Python versions."""
import numpy as np
import pytest

from pytfit5 import transitPy5 as tpy5


def light_curve(seed=1, n_days=12.0, cadence=15.0 / 1440.0, sigma=1e-3):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, n_days, cadence)
    keep = np.ones(t.size, bool)
    keep &= ~((t > 3.0) & (t < 3.8))            # a gap
    keep &= ~((t > 6.0) & (t < 6.9))            # a short segment isolated by two gaps:
    keep &= ~((t > 6.95) & (t < 8.0))           #   6.9-6.95 d (a handful of points)
    keep &= rng.random(t.size) > 0.03           # missing cadences
    t = t[keep]
    f = 1.0 + rng.normal(0, sigma, t.size) + 3e-3 * np.sin(2 * np.pi * t / 4.3)
    k = rng.choice(t.size, t.size // 100, replace=False)
    f[k] += rng.choice([-1, 1], k.size) * rng.uniform(5, 30, k.size) * sigma    # outliers
    dy = np.full(t.size, sigma) * rng.uniform(0.8, 1.2, t.size)
    tflag = np.zeros(t.size)
    for c, w in ((1.0, 0.1), (0.02, 0.1), (3.82, 0.1), (9.0, 0.6), (11.9, 0.2)):   # masked stretches
        tflag[np.abs(t - c) < w] = 1                                                # incl. edges, gap edge, wide
    return t, f, dy, tflag


@pytest.mark.parametrize("boxbin, nfitp", [(2.0, 3), (1.0, 3), (1.5, 2), (0.5, 1), (0.2, 3)])
@pytest.mark.parametrize("multipro", [0, 1])
def test_polyfilter_matches_reference(boxbin, nfitp, multipro):
    t, f, dy, tflag = light_curve()
    perm = np.random.default_rng(2).permutation(t.size)             # unsorted input
    args = (t[perm], f[perm], dy[perm], tflag[perm], boxbin, nfitp, 0.5)
    ref = tpy5.polyfilter_reference(*args, multipro=0)
    new = tpy5.polyfilter(*args, multipro=multipro)
    assert np.all(np.isfinite(new))
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-9)


def test_polyfilter_all_masked_window():
    """Every point near a long masked stretch still gets a finite trend."""
    t, f, dy, tflag = light_curve()
    tflag[(t > 8.0) & (t < 11.0)] = 1
    new = tpy5.polyfilter(t, f, dy, tflag, 0.5, 3, 0.5, multipro=0)
    ref = tpy5.polyfilter_reference(t, f, dy, tflag, 0.5, 3, 0.5, multipro=0)
    assert np.all(np.isfinite(new))
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-9)


@pytest.mark.parametrize("direction", ["both", "up"])
def test_cutoutliers_matches_reference(direction):
    t, f, _, _ = light_curve(seed=3)
    n = 6
    new = tpy5.cutoutliers(t, f, n, 3.0, direction)
    ref = tpy5.cutoutliers_reference(t, f, n, 3.0, direction)
    assert np.array_equal(new[n:-n], ref[n:-n])     # the reference used stale samples near the ends
    assert new.sum() > 0


def test_iterative_binned_matches_reference(monkeypatch):
    t, f, dy, _ = light_curve(seed=4)
    t0 = t[np.argmin(np.abs(t - 5.0))]
    f = f - 4e-3 * (np.abs(t - t0) < 0.08)           # a transit to be masked

    def run():
        p = tpy5.phot_class()
        p.time, p.flux, p.ferr = t.copy(), f.copy(), dy.copy()
        p.tflag = np.zeros(t.size); p.icut = np.zeros(t.size); p.flux_f = f.copy()
        inp = tpy5.tpy5_inputs_class()
        inp.multipro, inp.boxbin, inp.nfitp = 0, 1.0, 3
        inp.cut_direction, inp.iter_mask_mode, inp.iter_sigma_threshold = "up", "binned", 4.0
        tpy5.run_polyfilter_iterative(p, inp)
        tpy5.run_cutoutliers(p, inp)
        return p

    new = run()
    monkeypatch.setattr(tpy5, "polyfilter", tpy5.polyfilter_reference)
    ref = run()
    assert new.tflag[np.abs(t - t0) < 0.05].all()
    assert np.array_equal(new.tflag, ref.tflag)
    np.testing.assert_allclose(new.flux_f, ref.flux_f, rtol=0, atol=1e-9)
