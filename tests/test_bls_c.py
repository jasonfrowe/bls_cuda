"""The compiled C BLS kernel must match the Numba kernels exactly.

Run: python -m pytest tests/test_bls_c.py   (or python tests/test_bls_c.py)
Set PYTFIT5_TEST_LC=/path/to/lightcurve.dat to also check a real light curve.
"""
import os

import numpy as np
import pytest

from pytfit5 import bls_c, bls_cpu
from pytfit5.transitPy5 import tpy5_inputs_class

pytestmark = pytest.mark.skipif(not bls_c.available(), reason=f"C kernel not built: {bls_c.load_error()}")


def synthetic_lc(days=80.0, cadence_min=29.4, period=3.217, depth=5e-4, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, days, cadence_min / 1440.0)
    t = t[rng.random(t.size) > 0.05]                     # random gaps
    f = rng.normal(0.0, 2e-4, t.size)
    f[((t - 1.3) % period) < 0.12] -= depth             # box transits
    return t, f - np.median(f)


def grid(t, nbins=500):
    nb = min(nbins, t.size - 1)
    nyq = bls_cpu.nyquest_cpu(t)
    f1, f2, ofac = 2.0 / t.max(), 2.0, 8.0
    steps = int(ofac * (f2 - f1) * t.size / nyq)
    df0 = (f2 - f1) / steps
    n = bls_cpu.calc_nsteps_cpu(1.0, 1.0, f1, f2, nyq, ofac, t.size, df0, nb, 5)
    return bls_cpu.calc_freqs_cpu(n, 1.0, 1.0, f1, f2, nyq, ofac, t.size, df0, nb, 5), nb


def numba_loop(freqs, t, f, nbs):
    out = np.array([bls_cpu.bls_kernel(fr, t, f, nb, 1.0, 1.0) for fr, nb in zip(freqs, nbs)])
    return out[:, 0], out[:, 1].astype(np.int32), out[:, 2].astype(np.int32)


@pytest.mark.parametrize("variant", list(bls_c.VARIANTS))
def test_scan_matches_numba(variant):
    if variant not in bls_c.supported_variants():
        pytest.skip(f"{variant} not supported on this CPU")
    t, f = synthetic_lc()
    freqs, nb = grid(t)
    freqs = freqs[::7]
    nbs = np.full(freqs.size, nb, dtype=np.int64)
    nbs[::3] += 37                      # variable nb (max_bin_minutes path), odd sizes
    ref = numba_loop(freqs, t, f, nbs)
    for nthreads in (1, 0):
        got = bls_c.scan(freqs, nbs, t, f, isa=variant, nthreads=nthreads)
        for r, g in zip(ref, got):
            np.testing.assert_array_equal(g, r)


def test_small_nb_edge_cases():
    t, f = synthetic_lc(days=20.0)
    freqs, _ = grid(t)
    freqs = freqs[::50]
    for nb in (2, 3, 5, 31, 33, 64, 65):
        nbs = np.full(freqs.size, nb, dtype=np.int64)
        ref = numba_loop(freqs, t, f, nbs)
        got = bls_c.scan(freqs, nbs, t, f, isa="auto")
        for r, g in zip(ref, got):
            np.testing.assert_array_equal(g, r)


def _run_bls(t, f, engine, multipro):
    inp = tpy5_inputs_class()
    inp.plots = 0
    inp.multipro = multipro
    inp.nbins = 500
    inp.bls_engine = engine
    inp.return_spectrum = True
    return bls_cpu.bls(inp, time=t.copy(), flux=f.copy())


def test_bls_end_to_end_engines_agree():
    t, f = synthetic_lc()
    a = _run_bls(t, f, "numba", 0)
    b = _run_bls(t, f, "c", 1)
    assert a.engine == "numba" and b.engine.startswith("c:")
    np.testing.assert_array_equal(a.power, b.power)
    assert (a.bper, a.epo, a.tdur, a.depth, a.snr) == (b.bper, b.epo, b.tdur, b.depth, b.snr)
    assert abs(b.bper - 3.217) < 0.01


def test_unknown_isa_raises():
    with pytest.raises(ValueError):
        bls_c.resolve_variant("sse9")


@pytest.mark.skipif(not os.environ.get("PYTFIT5_TEST_LC"), reason="PYTFIT5_TEST_LC not set")
def test_real_lightcurve():
    d = np.loadtxt(os.environ["PYTFIT5_TEST_LC"], usecols=(0, 1))
    t, f = d[:, 0] - d[:, 0].min(), d[:, 1] - np.median(d[:, 1])
    freqs, nb = grid(t, nbins=2000)
    freqs = freqs[1.0 / freqs < t.max()][::97]
    nbs = np.full(freqs.size, nb, dtype=np.int64)
    ref = numba_loop(freqs, t, f, nbs)
    for v in bls_c.supported_variants():
        got = bls_c.scan(freqs, nbs, t, f, isa=v)
        for r, g in zip(ref, got):
            np.testing.assert_array_equal(g, r)


# ---------------------------------------------------------------- dips_only --
def numba_loop_dips(freqs, t, f, nbs, kernel=None):
    k = kernel or bls_cpu.bls_kernel
    out = np.array([k(fr, t, f, nb, 1.0, 1.0, 0.0) for fr, nb in zip(freqs, nbs)])
    return out[:, 0], out[:, 1].astype(np.int32), out[:, 2].astype(np.int32)


def test_dips_only_numba_matches_reference():
    t, f = synthetic_lc(depth=0.0, seed=7)          # noise only: brightenings compete
    freqs, nb = grid(t)
    freqs = freqs[::11]
    nbs = np.full(freqs.size, nb, dtype=np.int64)
    a = numba_loop_dips(freqs, t, f, nbs)
    b = numba_loop_dips(freqs, t, f, nbs, kernel=bls_cpu.bls_kernel_reference)
    for r, g in zip(b, a):
        np.testing.assert_array_equal(g, r)
    both = numba_loop(freqs, t, f, nbs)
    assert np.all(a[0] <= both[0]) and np.any(a[0] < both[0])


@pytest.mark.parametrize("variant", list(bls_c.VARIANTS))
def test_dips_only_scan_matches_numba(variant):
    if variant not in bls_c.supported_variants():
        pytest.skip(f"{variant} not supported on this CPU")
    t, f = synthetic_lc(depth=0.0, seed=7)
    freqs, nb = grid(t)
    freqs = freqs[::7]
    nbs = np.full(freqs.size, nb, dtype=np.int64)
    nbs[::3] += 37
    ref = numba_loop_dips(freqs, t, f, nbs)
    for nthreads in (1, 0):
        got = bls_c.scan(freqs, nbs, t, f, isa=variant, nthreads=nthreads, dips_only=True)
        for r, g in zip(ref, got):
            np.testing.assert_array_equal(g, r)


@pytest.mark.parametrize("engine,multipro", [("numba", 0), ("c", 1)])
def test_dips_only_end_to_end(engine, multipro):
    """With dips_only every top peak is a dip (positive depth); without it, about half are not."""
    neg = {False: 0, True: 0}
    for seed in range(8):
        t, f = synthetic_lc(depth=0.0, seed=100 + seed)
        for dips in (False, True):
            inp = tpy5_inputs_class()
            inp.plots, inp.multipro, inp.nbins, inp.bls_engine = 0, multipro, 500, engine
            inp.dips_only = dips
            a = bls_cpu.bls(inp, time=t.copy(), flux=f.copy())
            neg[dips] += a.depth < 0
    assert neg[True] == 0
    assert neg[False] > 0
    t, f = synthetic_lc()                            # a real transit is still found
    inp = tpy5_inputs_class()
    inp.plots, inp.multipro, inp.nbins, inp.bls_engine, inp.dips_only = 0, multipro, 500, engine, True
    assert abs(bls_cpu.bls(inp, time=t, flux=f).bper - 3.217) < 0.01


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
