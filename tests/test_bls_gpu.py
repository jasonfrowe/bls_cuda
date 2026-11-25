"""The CUDA BLS kernel (pytfit5/bls_gpu.py) against the CPU kernels.

The GPU bins every point exactly as the CPU kernels do, but searches the boxes in FP32,
so a box can differ from the CPU's only where two boxes have the same power to FP32
precision, and the power agrees to ~1e-7 relative. GPU tests are skipped without a CUDA
GPU; the fixed-point packing tests run anywhere.
"""
import numpy as np
import pytest

from pytfit5 import bls_cpu, bls_gpu
from pytfit5.transitPy5 import tpy5_inputs_class

needs_gpu = pytest.mark.skipif(not bls_gpu.available(), reason=f"no CUDA GPU: {bls_gpu.load_error()}")


def synthetic_lc(days=70.5, cadence_min=12.1, period=7.3, depth=8e-4, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, days, cadence_min / 1440.0)
    t = t[rng.random(t.size) > 0.05]                     # random gaps
    f = rng.normal(0.0, 1e-3, t.size)
    f[((t - 1.1) % period) < 0.12] -= depth
    return t, f - np.median(f)


def grid(t, nbins=1500, max_bin_minutes=5.0, ofac=4.0):
    """Frequencies and bins as bls_cpu.bls() makes them (Period < baseline only)."""
    nb = min(nbins, t.size - 1)
    nyq = bls_cpu.nyquest_cpu(t)
    f1, f2 = 2.0 / t.max(), 2.0
    df0 = (f2 - f1) / int(ofac * (f2 - f1) * t.size / nyq)
    n = bls_cpu.calc_nsteps_cpu(1.0, 1.0, f1, f2, nyq, ofac, t.size, df0, nb, 5)
    freqs = bls_cpu.calc_freqs_cpu(n, 1.0, 1.0, f1, f2, nyq, ofac, t.size, df0, nb, 5)
    nbs = np.full(n, nb, dtype=np.int64)
    if max_bin_minutes > 0:
        nbs = np.maximum(nbs, np.ceil(1440.0 / freqs / max_bin_minutes).astype(np.int64))
    keep = 1.0 / freqs <= t.max() - t.min()
    return freqs[keep], nbs[keep]


def cpu_scan(freqs, nbs, t, f, dips_only):
    """Reference: the C kernel if built, else the Numba kernel (bit-identical to each other)."""
    from pytfit5 import bls_c
    if bls_c.available():
        return bls_c.scan(freqs, nbs, t, f, dips_only=dips_only, nthreads=2)
    smax = 0.0 if dips_only else np.inf
    out = np.array([bls_cpu.bls_kernel(fr, t, f, nb, 1.0, 1.0, smax) for fr, nb in zip(freqs, nbs)])
    return out[:, 0], out[:, 1].astype(np.int32), out[:, 2].astype(np.int32)


def assert_agree(gpu, cpu):
    pg, j1g, j2g = gpu
    pc, j1c, j2c = cpu
    same = (j1g == j1c) & (j2g == j2c)
    assert same.mean() > 0.999, f"boxes differ at {np.sum(~same)} of {same.size} frequencies"
    np.testing.assert_allclose(pg[same], pc[same], rtol=1e-6, atol=0)
    # where the box differs, it must be a near-tie: the GPU's box has the CPU's power to 1e-6
    np.testing.assert_allclose(pg[~same], pc[~same], rtol=1e-6, atol=0)
    assert np.argmax(pg) == np.argmax(pc)


def test_pack_flux_sums_are_exact():
    rng = np.random.default_rng(0)
    for npt in (100, 70000):
        flux = rng.normal(0, 1e-3, npt)
        flux[:5] = [0.5, -0.5, 2.0, -1e-9, 0.0]
        packed, cb, inv = bls_gpu.pack_flux(flux)
        assert cb >= int(npt).bit_length()
        q = np.rint(flux / inv).astype(np.int64)
        assert np.max(np.abs(q * inv - flux)) <= 0.5 * inv          # quantisation only
        for sl in (slice(0, 7), slice(3, npt), slice(None)):       # any box: exact flux and count
            v = int(np.sum(packed[sl]))
            c = v & ((1 << cb) - 1)
            assert c == len(range(npt)[sl])
            assert (v - c) >> cb == int(np.sum(q[sl]))
        assert np.sum(np.abs(q)) < 2 ** (62 - cb)                    # no overflow possible


@needs_gpu
@pytest.mark.parametrize("dips_only", [False, True])
def test_scan_matches_cpu(dips_only):
    t, f = synthetic_lc()
    freqs, nbs = grid(t)
    assert_agree(bls_gpu.scan(freqs, nbs, t, f, dips_only=dips_only), cpu_scan(freqs, nbs, t, f, dips_only))


@needs_gpu
def test_global_memory_path_matches_cpu():
    """Long periods with many bins: force every frequency through the global-memory kernel."""
    t, f = synthetic_lc(days=40.0)
    freqs, nbs = grid(t, max_bin_minutes=2.0)
    freqs, nbs = freqs[::7], nbs[::7]
    assert_agree(bls_gpu.scan(freqs, nbs, t, f, max_shared_nb=1, global_mem_mb=4),
                 cpu_scan(freqs, nbs, t, f, False))


@needs_gpu
def test_scan_is_deterministic():
    t, f = synthetic_lc()
    freqs, nbs = grid(t)
    a = bls_gpu.scan(freqs, nbs, t, f)
    b = bls_gpu.scan(freqs, nbs, t, f)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@needs_gpu
def test_bls_engine_cuda():
    t, f = synthetic_lc()
    res = {}
    for engine in ("cuda", "c" if bls_cpu.bls_c.available() else "numba"):
        inp = tpy5_inputs_class()
        inp.ofac, inp.nbins, inp.max_bin_minutes, inp.dips_only = 4.0, 1500, 5.0, True
        inp.bls_engine, inp.multipro, inp.plots = engine, 0, 0
        res[engine] = bls_cpu.bls(inp, t.copy(), f.copy() + 1.0)
    g, c = res.pop("cuda"), next(iter(res.values()))
    assert g.engine == "cuda"
    assert g.bper == c.bper and g.epo == c.epo
    assert g.bpower == pytest.approx(c.bpower, rel=1e-6)
    assert g.snr == pytest.approx(c.snr, rel=1e-9)
