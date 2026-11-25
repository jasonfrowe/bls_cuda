"""Single-core timing of BLS kernels on a real Kepler light curve.

Usage:
    python avx512/bench.py [lcfile] [--stride N] [--kernels a,b,...] [--nbins 2000]

Builds the same frequency grid as bls_cpu.bls() (defaults: freq1=2/T, freq2=2 c/d,
ofac=8, Mstar=Rstar=1), runs every Nth frequency through each kernel, reports
wall time and checks (p, jn1, jn2) against bls_kernel_reference.
Run under `taskset -c <core>` for stable numbers.
"""
import argparse
import time as _time
import importlib

import numpy as np
from pytfit5 import bls_cpu

DEFAULT_LC = "/data/Kepler/Kepler_n/koi1.n/klc11446443.dc.dat"


def load_lc(path):
    d = np.loadtxt(path, usecols=(0, 1))
    t, f = d[:, 0].copy(), d[:, 1].copy()
    t -= t.min()
    f -= np.median(f)
    return t, f


def freq_grid(t, ofac=8.0, Mstar=1.0, Rstar=1.0, nbins=2000, freq1=-1, freq2=-1, minbin=5):
    npt = t.size
    nb = min(nbins, npt - 1)
    nyq = bls_cpu.nyquest_cpu(t)
    if freq1 <= 0:
        freq1 = 2.0 / (t.max() - t.min())
    if freq2 <= 0:
        freq2 = 2.0
    steps = int(ofac * (freq2 - freq1) * npt / nyq)
    df0 = (freq2 - freq1) / steps
    nstep = bls_cpu.calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin)
    return bls_cpu.calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, ofac, npt, df0, nb, minbin), nb


def get_kernel(name):
    """name is 'func' (from bls_cpu) or 'module:func'."""
    if ":" in name:
        mod, fn = name.split(":")
        return getattr(importlib.import_module(mod), fn)
    return getattr(bls_cpu, name)


def run(kernel, freqs, t, f, nb):
    out = np.empty((freqs.size, 3))
    t0 = _time.perf_counter()
    for i, fr in enumerate(freqs):
        out[i] = kernel(fr, t, f, nb, 1.0, 1.0)
    return _time.perf_counter() - t0, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lc", nargs="?", default=DEFAULT_LC)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--nbins", type=int, default=2000)
    ap.add_argument("--kernels", default="bls_kernel_reference,bls_kernel")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    t, f = load_lc(a.lc)
    freqs, nb = freq_grid(t, nbins=a.nbins)
    T = t.max()
    freqs_all = freqs
    freqs = freqs[1.0 / freqs < T][:: a.stride]  # bls_kernel path only
    print(f"lc={a.lc} npt={t.size} nstep={freqs_all.size} sampled={freqs.size} nb={nb}")

    ref = None
    for name in a.kernels.split(","):
        k = get_kernel(name)
        k(freqs[0], t, f, nb, 1.0, 1.0)  # compile / warm
        best = min(run(k, freqs, t, f, nb)[0] for _ in range(a.repeat))
        _, out = run(k, freqs, t, f, nb)
        if ref is None:
            ref = out
            ref_time = best
        exact = np.array_equal(out, ref)
        relp = np.max(np.abs(out[:, 0] - ref[:, 0]) / np.maximum(ref[:, 0], 1e-300))
        idx_mismatch = int(np.sum(np.any(out[:, 1:] != ref[:, 1:], axis=1)))
        print(f"{name:40s} {best:8.3f} s  x{ref_time / best:5.2f}  "
              f"per-freq {1e6 * best / freqs.size:8.1f} us  exact={exact} "
              f"max_rel_dp={relp:.1e} idx_mismatch={idx_mismatch}")
        # Projected full-scan single-core time
        print(f"{'':40s} projected full scan (1 core): {best * a.stride:7.1f} s")
