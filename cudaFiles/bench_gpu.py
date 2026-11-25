"""GPU (pytfit5.bls_gpu) vs C (pytfit5.bls_c) BLS kernel on a simulated Roman GBTDS light curve.

Production grid by default (ofac 4, nb >= 1500, bins <= 5 min). Reports the kernel times, the
fraction of frequencies where both kernels pick the same box, and the power difference.

  python cudaFiles/bench_gpu.py --nseasons 2 --dips --threads 8
  cudaFiles/gpurun.sh gpu.log python cudaFiles/bench_gpu.py      # with GPU temperature logging
"""
import argparse, time as _t
import numpy as np
from pytfit5 import bls_cpu, bls_c, bls_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--nseasons", type=int, default=1)
ap.add_argument("--ofac", type=float, default=4.0)
ap.add_argument("--nbins", type=int, default=1500)
ap.add_argument("--maxbin", type=float, default=5.0)
ap.add_argument("--dips", action="store_true")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--noc", action="store_true", help="skip the C kernel")
a = ap.parse_args()

rng = np.random.default_rng(1)
cad = 12.1 / 1440
t = np.concatenate([np.arange(0, 70.5, cad) + s * 182.0 for s in range(a.nseasons)])
f = 1 + 1e-3 * rng.standard_normal(t.size)
per, t0, dur = 7.3, 1.1, 0.12
f[np.abs((t - t0 + 0.5 * per) % per - 0.5 * per) < dur / 2] -= 8e-4
t = t - t.min(); f = f - np.median(f)
Mstar = Rstar = 1.0
npt = t.size
nb = min(a.nbins, npt - 1)
nyq = bls_cpu.nyquest_cpu(t)
freq1, freq2 = 2.0 / (t.max() - t.min()), 2.0
steps = int(a.ofac * (freq2 - freq1) * npt / nyq)
df0 = (freq2 - freq1) / steps
nstep = bls_cpu.calc_nsteps_cpu(Mstar, Rstar, freq1, freq2, nyq, a.ofac, npt, df0, nb, 5)
freqs = bls_cpu.calc_freqs_cpu(nstep, Mstar, Rstar, freq1, freq2, nyq, a.ofac, npt, df0, nb, 5)
nbs = np.full(nstep, nb, dtype=np.int64)
if a.maxbin > 0:
    nbs = np.maximum(nbs, np.ceil((1.0 / freqs) * 1440.0 / a.maxbin).astype(np.int64))
multi = (1.0 / freqs) <= (t.max() - t.min())
freqs, nbs = freqs[multi], nbs[multi]
print(f"npt {npt}  nfreq {freqs.size}  nb {nbs.min()}..{nbs.max()}  max_nb(GPU) {bls_gpu.max_nb()}"
      f"  frac nb>max_nb {np.mean(nbs > bls_gpu.max_nb()):.4f}")

tg = []
for r in range(a.reps):
    s = _t.perf_counter()
    pg, j1g, j2g = bls_gpu.scan(freqs, nbs, t, f, Rstar, Mstar, dips_only=a.dips)
    tg.append(_t.perf_counter() - s)
print("GPU s:", " ".join(f"{x:.3f}" for x in tg), " (the first includes compiling)")
if a.noc:
    raise SystemExit
s = _t.perf_counter()
pc, j1c, j2c = bls_c.scan(freqs, nbs, t, f, Rstar, Mstar, nthreads=a.threads, dips_only=a.dips)
tc = _t.perf_counter() - s
print(f"C ({bls_c.resolve_variant()}, {a.threads} threads) s: {tc:.3f}  -> x{a.threads} core-equivalent {tc*a.threads:.2f} core-s")
rel = np.abs(pg - pc) / np.maximum(pc, 1e-300)
same = (j1g == j1c) & (j2g == j2c)
print(f"(jn1,jn2) identical: {same.mean():.6f}   p rel diff: median {np.median(rel):.2e} max {rel.max():.2e}")
for i in np.flatnonzero(~same)[:5]:
    print(f"  f={freqs[i]:.6f} nb={nbs[i]} C p={pc[i]:.6e} ({j1c[i]},{j2c[i]})  GPU p={pg[i]:.6e} ({j1g[i]},{j2g[i]})")
ib = np.argmax(pc); ig = np.argmax(pg)
print(f"top peak: C P={1/freqs[ib]:.5f} ({j1c[ib]},{j2c[ib]})   GPU P={1/freqs[ig]:.5f} ({j1g[ig]},{j2g[ig]})")
