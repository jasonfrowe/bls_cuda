"""Multi-threaded full-scan comparison: C/OpenMP variants vs Numba bls_kernel in a thread pool.

Usage: OMP_NUM_THREADS=N python avx512/scan_bench.py [lc] [--variants ...]
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load_lc, freq_grid, DEFAULT_LC
import cbls

ap = argparse.ArgumentParser()
ap.add_argument("lc", nargs="?", default=DEFAULT_LC)
ap.add_argument("--variants", default="avx2_div,avx2_nodiv,avx512_div,avx512_nodiv")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--ref", default="avx2_div", help="variant used as reference (bit-identical to bls_kernel)")
a = ap.parse_args()

t, f = load_lc(a.lc)
freqs, nb = freq_grid(t)
freqs = freqs[1.0 / freqs < t.max()][:: a.stride]
nbs = np.full(freqs.size, nb, dtype=np.int64)
print(f"npt={t.size} nfreq={freqs.size} threads={os.environ.get('OMP_NUM_THREADS', 'all')}")
cbls.scan(freqs[:2000], nbs[:2000], t, f, variant="scalar")  # warm threads
ref = None
for v in [a.ref] + [x for x in a.variants.split(",") if x != a.ref]:
    t0 = time.perf_counter()
    p, j1, j2 = cbls.scan(freqs, nbs, t, f, variant=v)
    dt = time.perf_counter() - t0
    if ref is None:
        ref, rt = (p, j1, j2), dt
    ex = np.array_equal(p, ref[0]) and np.array_equal(j1, ref[1]) and np.array_equal(j2, ref[2])
    nmis = int(np.sum((j1 != ref[1]) | (j2 != ref[2])))
    print(f"{v:14s} {dt:7.2f} s  x{rt/dt:5.2f}  exact={ex} idx_mismatch={nmis} "
          f"best P={1/freqs[np.argmax(p)]:.6f} d")
