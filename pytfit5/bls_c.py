"""ctypes interface to the compiled C/OpenMP BLS kernel (pytfit5/_cbls.c).

The extension is optional: if it was not built, ``available()`` returns False
and bls_cpu.bls() falls back to the Numba kernels.

Variants (``isa`` argument):
    "auto"          best supported: avx512_nodiv > avx2_div > scalar
    "avx512"        alias for avx512_nodiv
    "avx2"          alias for avx2_div
    "scalar", "avx2_div", "avx2_nodiv", "avx512_div", "avx512_nodiv"
All variants return (p, jn1, jn2) bit-identical to bls_cpu.bls_kernel.
"""
import ctypes
import importlib.util
import os

import numpy as np

VARIANTS = {"scalar": 0, "avx2_div": 1, "avx2_nodiv": 2, "avx512_div": 3, "avx512_nodiv": 4}
ALIASES = {"avx512": "avx512_nodiv", "avx2": "avx2_div"}
AUTO_ORDER = ("avx512_nodiv", "avx2_div", "scalar")

_lib = None
_load_error = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        path = os.environ.get("PYTFIT5_CBLS_LIB")
        if not path:
            spec = importlib.util.find_spec("pytfit5._cbls")
            if spec is None or spec.origin is None:
                raise ImportError("pytfit5._cbls extension not built")
            path = spec.origin
        lib = ctypes.CDLL(path)
        dp = ctypes.POINTER(ctypes.c_double)
        i32p = ctypes.POINTER(ctypes.c_int32)
        lib.cbls_scan.argtypes = [dp, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, dp, dp,
                                  ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                  ctypes.c_int, dp, i32p, i32p]
        lib.cbls_scan.restype = ctypes.c_int
        if hasattr(lib, "cbls_scan_ex"):   # absent in builds older than dips_only
            lib.cbls_scan_ex.argtypes = [dp, ctypes.POINTER(ctypes.c_int64), ctypes.c_int, dp, dp,
                                         ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_double, dp, i32p, i32p]
            lib.cbls_scan_ex.restype = ctypes.c_int
        lib.cbls_supported_variants.restype = ctypes.c_int
        lib.cbls_has_openmp.restype = ctypes.c_int
        _lib = lib
    except Exception as exc:  # noqa: BLE001 - any failure means "use Numba"
        _load_error = exc
    return _lib


def available():
    """True if the compiled kernel could be loaded."""
    return _load() is not None


def load_error():
    """Why the kernel is unavailable (None if it loaded)."""
    _load()
    return _load_error


def has_openmp():
    return available() and bool(_lib.cbls_has_openmp())


def supported_variants():
    """Variant names this CPU can run (empty if the library is unavailable)."""
    if not available():
        return []
    mask = _lib.cbls_supported_variants()
    return [n for n, v in VARIANTS.items() if (mask >> v) & 1]


def resolve_variant(isa="auto"):
    """Map an isa request to a concrete, supported variant name (raises if impossible)."""
    sup = supported_variants()
    if not sup:
        raise RuntimeError(f"C BLS kernel unavailable: {load_error()}")
    isa = (isa or "auto").lower()
    if isa == "auto":
        return next(v for v in AUTO_ORDER if v in sup)
    name = ALIASES.get(isa, isa)
    if name not in VARIANTS:
        raise ValueError(f"unknown bls_isa {isa!r}; choose auto, "
                         f"{', '.join(list(ALIASES) + list(VARIANTS))}")
    if name not in sup:
        raise RuntimeError(f"bls_isa {isa!r} not supported on this CPU (supported: {sup})")
    return name


def scan(freqs, nbs, time, flux, Rstar=1.0, Mstar=1.0, isa="auto", nthreads=0, dips_only=False):
    """BLS over all frequencies (Period < baseline path of bls_cpu.bls_kernel).

    nthreads <= 0 uses OpenMP's default (OMP_NUM_THREADS or all CPUs).
    dips_only=True counts only boxes whose (median-centred) flux sum is negative,
    i.e. transits; False keeps both signs (classic s^2 BLS).
    Returns p (float64), jn1, jn2 (int32), same as looping bls_kernel.
    """
    variant = resolve_variant(isa)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    nbs = np.ascontiguousarray(np.broadcast_to(nbs, freqs.shape), dtype=np.int64)
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    p = np.zeros(freqs.size)
    jn1 = np.zeros(freqs.size, dtype=np.int32)
    jn2 = np.zeros(freqs.size, dtype=np.int32)
    dp = ctypes.POINTER(ctypes.c_double)
    i32p = ctypes.POINTER(ctypes.c_int32)
    args = (freqs.ctypes.data_as(dp), nbs.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            freqs.size, time.ctypes.data_as(dp), flux.ctypes.data_as(dp), time.size,
            float(Rstar), float(Mstar), VARIANTS[variant], int(nthreads))
    out = (p.ctypes.data_as(dp), jn1.ctypes.data_as(i32p), jn2.ctypes.data_as(i32p))
    if hasattr(_lib, "cbls_scan_ex"):
        rc = _lib.cbls_scan_ex(*args, 0.0 if dips_only else np.inf, *out)
    elif dips_only:
        raise RuntimeError("the built C kernel predates dips_only; rebuild pytfit5 (pip install -e .)")
    else:
        rc = _lib.cbls_scan(*args, *out)
    if rc != 0:
        raise RuntimeError(f"cbls_scan failed (code {rc})")
    return p, jn1, jn2
