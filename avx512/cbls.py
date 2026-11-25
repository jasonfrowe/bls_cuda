"""ctypes wrapper for libcbls.so (build with avx512/build.sh)."""
import ctypes
import os
import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcbls.so"))
_dp = ctypes.POINTER(ctypes.c_double)
_lib.bls_kernel_c.argtypes = [ctypes.c_double, _dp, _dp, ctypes.c_int, ctypes.c_int,
                              ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, _dp]
_lib.bls_scan_c.argtypes = [_dp, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, _dp, _dp,
                            ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                            _dp, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]

VARIANTS = {"scalar": 0, "avx2_div": 1, "avx2_nodiv": 2, "avx512_div": 3, "avx512_nodiv": 4}
# "+pb" suffix: interleaved-pair binning (scan only)
VARIANTS.update({k + "+pb": v | 16 for k, v in list(VARIANTS.items())})


def _ptr(a, t=_dp):
    return a.ctypes.data_as(t)


def _make(variant, mode=0):
    out = np.zeros(3)

    def k(freq, time, flux, nb, Rstar, Mstar):
        time = np.ascontiguousarray(time, dtype=np.float64)
        flux = np.ascontiguousarray(flux, dtype=np.float64)
        _lib.bls_kernel_c(freq, _ptr(time), _ptr(flux), time.size, int(nb), Rstar, Mstar,
                          variant, mode, _ptr(out))
        return out[0], int(out[1]), int(out[2])
    return k


for _n, _v in [(k, v) for k, v in VARIANTS.items() if v < 16]:
    globals()[_n] = _make(_v)
    globals()[_n + "_search"] = _make(_v, 2)
bin_only = _make(0, 1)


def scan(freqs, nbs, time, flux, Rstar=1.0, Mstar=1.0, variant="avx2_div"):
    """OpenMP scan over all frequencies; set OMP_NUM_THREADS to control threads."""
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    nbs = np.ascontiguousarray(nbs, dtype=np.int64)
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    p = np.zeros(freqs.size)
    jn1 = np.zeros(freqs.size, dtype=np.int32)
    jn2 = np.zeros(freqs.size, dtype=np.int32)
    _lib.bls_scan_c(_ptr(freqs), _ptr(nbs, ctypes.POINTER(ctypes.c_int64)), freqs.size,
                    _ptr(time), _ptr(flux), time.size, Rstar, Mstar, VARIANTS[variant],
                    _ptr(p), _ptr(jn1, ctypes.POINTER(ctypes.c_int32)),
                    _ptr(jn2, ctypes.POINTER(ctypes.c_int32)))
    return p, jn1, jn2
