"""End-to-end bls() timing: Numba engine vs C engine (each isa given), plots off.

Usage: python avx512/bls_e2e.py [lcfile] [isa ...]
"""
import sys, time
import numpy as np
from pytfit5 import bls_cpu
from pytfit5.transitPy5 import tpy5_inputs_class


def run(t, f, engine, isa="auto"):
    inp = tpy5_inputs_class()
    inp.plots = 0
    inp.bls_engine = engine
    inp.bls_isa = isa
    inp.return_spectrum = True
    t0 = time.perf_counter()
    a = bls_cpu.bls(inp, time=t.copy(), flux=f.copy())
    return time.perf_counter() - t0, a


if __name__ == "__main__":
    lc = sys.argv[1] if len(sys.argv) > 1 else "/data/Kepler/Kepler_n/koi1.n/klc11446443.dc.dat"
    isas = sys.argv[2:] or ["auto"]
    d = np.loadtxt(lc, usecols=(0, 1))
    t, f = d[:, 0], d[:, 1]
    tn, a = run(t, f, "numba")
    print(f"{lc}  npt={t.size}\n  numba            {tn:7.2f} s  P={a.bper:.6f} snr={a.snr:.2f}")
    for isa in isas:
        tc, b = run(t, f, "c", isa)
        same = np.array_equal(a.power, b.power) and (a.bper, a.epo, a.snr) == (b.bper, b.epo, b.snr)
        print(f"  {b.engine:16s} {tc:7.2f} s  x{tn/tc:.2f}  identical={same}")
