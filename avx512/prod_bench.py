"""Time the current production path (bls_cpu.compute_bls_kernel in a ProcessPool, as in
bls_cpu.bls) against cbls.scan on the same frequency grid, and compare results."""
import concurrent.futures, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load_lc, freq_grid, DEFAULT_LC
from pytfit5 import bls_cpu
import cbls

def main():
    lc = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LC
    variant = sys.argv[2] if len(sys.argv) > 2 else "avx512_nodiv+pb"
    t, f = load_lc(lc)
    freqs, nb = freq_grid(t)
    freqs = freqs[1.0 / freqs < t.max()]
    nbs = np.full(freqs.size, nb, dtype=np.int64)
    nstep = freqs.size
    nproc = len(os.sched_getaffinity(0))
    bls_cpu.bls_kernel(freqs[0], t, f, nb, 1.0, 1.0)

    t0 = time.perf_counter()
    ndiv = nstep // nproc + 1
    iarg = np.zeros((nproc, ndiv), dtype=np.int32)
    for i in range(nproc):
        for k, j in enumerate(range(i, nstep, nproc)):
            iarg[i, k] = j
    p = np.zeros((nproc, ndiv)); j1 = np.zeros((nproc, ndiv), np.int32); j2 = np.zeros((nproc, ndiv), np.int32)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(bls_cpu.compute_bls_kernel, iarg[i], freqs, t, f, nbs, 1.0, 1.0, ndiv): i for i in range(nproc)}
        for fu in concurrent.futures.as_completed(futs):
            i = futs[fu]; p[i], j1[i], j2[i] = fu.result()
    p = p.T.ravel()[:nstep]; j1 = j1.T.ravel()[:nstep]; j2 = j2.T.ravel()[:nstep]
    tn = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc, c1, c2 = cbls.scan(freqs, nbs, t, f, variant=variant)
    tc = time.perf_counter() - t0
    ex = np.array_equal(p, pc) and np.array_equal(j1, c1) and np.array_equal(j2, c2)
    print(f"{os.path.basename(lc)} npt={t.size} nfreq={nstep} procs={nproc}")
    print(f"  numba ProcessPool : {tn:7.2f} s")
    print(f"  cbls {variant:14s}: {tc:7.2f} s   speedup x{tn/tc:.2f}  exact={ex} "
          f"idx_mismatch={int(np.sum((j1 != c1) | (j2 != c2)))} maxrel={np.max(np.abs(p-pc)/np.maximum(p,1e-300)):.1e}")


if __name__ == "__main__":
    main()
