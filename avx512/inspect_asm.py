"""Report which SIMD register widths Numba emits for the BLS kernels.

Usage:  python avx512/inspect_asm.py
Set NUMBA_OPT / NUMBA_CPU_NAME / NUMBA_CPU_FEATURES in the environment to compare targets.
"""
import re
import numpy as np
from pytfit5 import bls_cpu


def summarize(fn, name):
    sig = fn.signatures[0]
    asm = fn.inspect_asm(sig)
    xmm = len(re.findall(r"%xmm\d+", asm))
    ymm = len(re.findall(r"%ymm\d+", asm))
    zmm = len(re.findall(r"%zmm\d+", asm))
    divs = {k: len(re.findall(k, asm)) for k in ("vdivpd", "vdivsd", "vdivps")}
    kmask = len(re.findall(r"%k[1-7]", asm))
    print(f"{name:22s} xmm={xmm:5d} ymm={ymm:5d} zmm={zmm:5d} kmask={kmask:4d} {divs}")
    return asm


if __name__ == "__main__":
    t = np.sort(np.random.default_rng(0).uniform(0, 1400, 60000))
    f = np.random.default_rng(1).normal(0, 1e-4, t.size)
    bls_cpu.bls_kernel(0.3, t, f, 2000, 1.0, 1.0)
    bls_cpu.bls_kernel_reference(0.3, t, f, 2000, 1.0, 1.0)
    asm = summarize(bls_cpu.bls_kernel, "bls_kernel")
    summarize(bls_cpu.bls_kernel_reference, "bls_kernel_reference")
    open("/tmp/bls_kernel.s", "w").write(asm)
