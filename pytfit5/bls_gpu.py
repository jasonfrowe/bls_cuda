"""CUDA BLS kernel (numba-cuda) for the Period < baseline frequencies of bls_cpu.bls().

Same search as bls_cpu.bls_kernel and the C kernel (pytfit5/_cbls.c): the same phase
bins, the same box-width limits and the same tie rule (first start bin wins). The
arithmetic differs, because FP64 runs at 1/32-1/64 of the FP32 rate on most NVIDIA
cards:

  phase      FP64, same operations as the CPU kernels, so every point lands in the
             same bin;
  binning    one int64 per point: the flux as a fixed-point integer in the high bits,
             a count of 1 in the low ``cb`` bits. One shared-memory integer atomic per
             point bins flux and count together, exactly and in any order, so results
             do not depend on thread scheduling;
  search     FP32 running sums of the bins, with the division-free comparison of the
             C kernel (s^2 D_best > N_best dfac);
  power      the winning box's power recomputed in FP64 from the FP32 bins.

So (jn1, jn2) agree with the CPU except where two boxes have powers equal to FP32
precision (~1e-7), and p agrees to ~1e-6 relative. Results are deterministic.

One thread block searches one frequency. Its bins live in shared memory (8 bytes per
bin, int64 during binning and then two float32 in place) when nb <= ``max_nb()``
(about 12,000 on a GPU with 100 KB of shared memory per block), otherwise in global
memory.
"""
import math
import warnings

import numpy as np

try:
    from numba import cuda, int32, int64, uint64, float32, float64, complex64
    _CUDA_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - numba-cuda missing or broken
    cuda = None
    _CUDA_IMPORT_ERROR = exc

G_CONST = 6.674e-11
RSUN = 696265.0e0 * 1000.0e0
MSUN = 1.9891e30
PIFAC = 1.083852140278e0
DAY2SEC = 86400.0
ONEHOUR = 1.0 / 24.0

THREADS = 256          # threads per block (one block per frequency)
_RED_BYTES = THREADS * (8 + 4 + 4)   # static shared memory for the block reduction


def available():
    """True if numba-cuda imported and a CUDA GPU is visible."""
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def load_error():
    if cuda is None:
        return _CUDA_IMPORT_ERROR
    return None if available() else RuntimeError("no CUDA GPU visible")


def max_nb():
    """Largest number of phase bins one block can hold in shared memory."""
    dev = cuda.get_current_device()
    smem = getattr(dev, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", 0) or dev.MAX_SHARED_MEMORY_PER_BLOCK
    return int((smem - _RED_BYTES - 1024) // 8)


def limits(freqs, nbs, npt, Rstar, Mstar):
    """kmi, kma, kkmi per frequency, with the operations of bls_kernel / _cbls.c limits()."""
    fsec = freqs / DAY2SEC
    q = PIFAC * Rstar * RSUN / (G_CONST * Mstar * MSUN) ** (1.0 / 3.0) * fsec ** (2.0 / 3.0)
    qmi = np.maximum(q / 2.0, ONEHOUR * freqs / 2)
    qma = np.minimum(q * 2.0, 0.25)
    swap = qmi > qma
    qma = np.where(swap, np.minimum(qmi * 2.0, 0.4), qma)
    kmi = np.maximum((qmi * nbs).astype(np.int64), 1)
    kma = (qma * nbs).astype(np.int64) + 1
    kkmi = np.maximum((npt * qmi).astype(np.int64), 5)
    return kmi.astype(np.int32), kma.astype(np.int32), kkmi.astype(np.int32)


def pack_flux(flux):
    """int64 per point: round(flux 2^e) << cb, plus 1 (the count).

    cb holds any box count (>= bit length of npt); e is the largest power of two
    (at most 40) for which the sum of |flux| over all points still fits, so no bin
    or box sum can overflow. Returns packed, cb, 2^-e.
    """
    flux = np.asarray(flux, dtype=np.float64)
    npt = flux.size
    cb = max(16, int(npt).bit_length())
    tot = float(np.sum(np.abs(flux)))
    e = 40 if tot == 0 else min(40, math.floor(math.log2(2.0 ** (61 - cb) / tot)))
    q = np.rint(flux * 2.0 ** e).astype(np.int64)
    return (q << cb) + 1, cb, 2.0 ** (-e)


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _bin_fp64(t, freq, fnbm1):
        """The CPU kernels' bin: trunc(frac(t f) (nb - 1)), with the same FP64 operations.
        (A double-float FP32 version with an FP64 recheck near bin edges was slower on an
        RTX PRO 4000 Blackwell: binning is limited by loading the points, not by FP64.)"""
        tf = t * freq
        ph = tf - float64(int32(tf))
        return int32(ph * fnbm1)

    # The two loops below index fc with an unsigned counter and use while loops: with a
    # signed index Numba adds a negative-index wraparound and range-iterator bookkeeping
    # to every step, about doubling the instructions per box.
    @cuda.jit(device=True, inline=True)
    def _accum(fc, j, j1, s, kk):
        """Add bins fc[j:j1] to (s, kk)."""
        while j < j1:
            v = fc[j]
            s += v.real
            kk += v.imag
            j += uint64(1)
        return s, kk

    @cuda.jit(device=True, inline=True)
    def _boxes(fc, j, j1, m, s, kk, BN, BD, BE, fnpt, fkkmi, smax):
        """Boxes ending at bins fc[j:j1]; the first has width index m."""
        while j < j1:
            v = fc[j]
            s += v.real
            kk += v.imag
            dfac = kk * (fnpt - kk)
            s2 = s * s
            if kk > fkkmi and dfac > 0.0 and s < smax and s2 * BD > BN * dfac:
                BN = s2
                BD = dfac
                BE = m
            j += uint64(1)
            m += 1
        return s, kk, BN, BD, BE

    @cuda.jit(device=True)
    def _search_one(k, bins, fc, red_p, red_i, red_m, time, packed, freqs, nbs, kmis,
                    kmas, kkmis, npt, cb, inv_scale, smax, p_out, jn1_out, jn2_out):
        """One block searches frequency k. bins (int64, nb) and fc (complex64, nb: flux sum,
        count) may be the same memory: each thread overwrites only the bins it has read."""
        tx = cuda.threadIdx.x
        bd = cuda.blockDim.x

        nb = int32(nbs[k])
        freq = freqs[k]
        for j in range(tx, nb, bd):
            bins[j] = 0
        cuda.syncthreads()

        # phase bins: same FP64 operations as the CPU kernels
        fnbm1 = float64(nb - 1)
        for i in range(tx, npt, bd):
            cuda.atomic.add(bins, _bin_fp64(time[i], freq, fnbm1), packed[i])
        cuda.syncthreads()

        # unpack each bin in place: fc[j] = (flux sum, count), read as one 64-bit load
        mask = (int64(1) << cb) - 1
        for j in range(tx, nb, bd):
            v = bins[j]
            c = v & mask
            y = (v - c) >> cb
            fc[j] = complex64(complex(float32(float64(y) * inv_scale), float32(c)))
        cuda.syncthreads()

        kmi = kmis[k]
        kma = kmas[k]
        fnpt = float32(npt)
        fkkmi = float32(kkmis[k])
        nacc = kmi if kmi < kma + 1 else kma + 1

        bestN = float32(0.0)
        bestD = float32(1.0)
        besti = int32(-1)
        bestm = int32(0)
        for i in range(tx, nb, bd):
            # bins i + m, wrapping at nb: m < nb - i reads fc[i + m], the rest fc[i + m - nb]
            mw = nb - i
            s = float32(0.0)
            kk = float32(0.0)
            # bin index i + m for m < mw, i + m - nb after; j = start, j1 = end (exclusive)
            ma = min(nacc, mw)
            s, kk = _accum(fc, uint64(i), uint64(i + ma), s, kk)
            if nacc > mw:
                s, kk = _accum(fc, uint64(0), uint64(i + nacc - nb), s, kk)
            BN = float32(0.0)
            BD = float32(1.0)
            BE = int64(-1)
            mb = min(kma + 1, mw)
            if mb > nacc:
                s, kk, BN, BD, BE = _boxes(fc, uint64(i + nacc), uint64(i + mb), int64(nacc),
                                           s, kk, BN, BD, BE, fnpt, fkkmi, smax)
            m0 = max(nacc, mw)
            if kma + 1 > m0:
                s, kk, BN, BD, BE = _boxes(fc, uint64(i + m0 - nb), uint64(i + kma + 1 - nb), int64(m0),
                                           s, kk, BN, BD, BE, fnpt, fkkmi, smax)
            # start bins are visited in increasing order: strict > keeps the first best
            if BE >= 0 and BN * bestD > bestN * BD:
                bestN = BN
                bestD = BD
                besti = i
                bestm = BE

        red_p[tx] = bestN / bestD if besti >= 0 else -1.0
        red_i[tx] = besti
        red_m[tx] = bestm
        cuda.syncthreads()
        stride = bd // 2
        while stride > 0:
            if tx < stride:
                po = red_p[tx + stride]
                io = red_i[tx + stride]
                pm = red_p[tx]
                if po > pm or (po == pm and io >= 0 and (red_i[tx] < 0 or io < red_i[tx])):
                    red_p[tx] = po
                    red_i[tx] = io
                    red_m[tx] = red_m[tx + stride]
            cuda.syncthreads()
            stride //= 2

        wi = red_i[0]
        wm = red_m[0]
        cuda.syncthreads()
        if wi < 0:
            if tx == 0:
                p_out[k] = 0.0
                jn1_out[k] = 0
                jn2_out[k] = 0
            return

        # power of the winning box in FP64 (deterministic tree sum over its bins)
        sy = float64(0.0)
        sc = float64(0.0)
        for m in range(tx, wm + 1, bd):
            j = wi + m
            if j >= nb:
                j -= nb
            v = fc[j]
            sy += float64(v.real)
            sc += float64(v.imag)
        red_p[tx] = sy
        cuda.syncthreads()
        stride = bd // 2
        while stride > 0:
            if tx < stride:
                red_p[tx] += red_p[tx + stride]
            cuda.syncthreads()
            stride //= 2
        sy = red_p[0]
        cuda.syncthreads()
        red_p[tx] = sc
        cuda.syncthreads()
        stride = bd // 2
        while stride > 0:
            if tx < stride:
                red_p[tx] += red_p[tx + stride]
            cuda.syncthreads()
            stride //= 2
        if tx == 0:
            sc = red_p[0]
            p_out[k] = sy * sy / (sc * (float64(npt) - sc))
            jn1_out[k] = wi
            jn2_out[k] = wi + wm


    @cuda.jit(cache=True)
    def _bls_shared(time, packed, freqs, nbs, kmis, kmas, kkmis, idx, npt, cb,
                    inv_scale, smax, p_out, jn1_out, jn2_out):
        """Bins in shared memory (nb <= max_nb())."""
        bins = cuda.shared.array(0, dtype=int64)     # nb bins, int64 while binning
        fc = cuda.shared.array(0, dtype=complex64)   # same memory: (flux sum, count) pairs
        red_p = cuda.shared.array(THREADS, dtype=float64)
        red_i = cuda.shared.array(THREADS, dtype=int32)
        red_m = cuda.shared.array(THREADS, dtype=int32)
        _search_one(idx[cuda.blockIdx.x], bins, fc, red_p, red_i, red_m, time, packed,
                    freqs, nbs, kmis, kmas, kkmis, npt, cb, inv_scale, smax, p_out, jn1_out, jn2_out)

    @cuda.jit(cache=True)
    def _bls_global(time, packed, freqs, nbs, kmis, kmas, kkmis, idx, npt, cb,
                    inv_scale, smax, p_out, jn1_out, jn2_out, gbins, gfc):
        """Bins in global memory, one row of gbins/gfc per block (any nb)."""
        b = cuda.blockIdx.x
        red_p = cuda.shared.array(THREADS, dtype=float64)
        red_i = cuda.shared.array(THREADS, dtype=int32)
        red_m = cuda.shared.array(THREADS, dtype=int32)
        _search_one(idx[b], gbins[b], gfc[b], red_p, red_i, red_m, time, packed,
                    freqs, nbs, kmis, kmas, kkmis, npt, cb, inv_scale, smax, p_out, jn1_out, jn2_out)


_smem_allowed = {}


def _allow_smem(nbytes, args):
    """Opt the kernel in to more than 48 KB of dynamic shared memory per block."""
    dev = cuda.get_current_device().id
    if _smem_allowed.get(dev, 0) >= nbytes:
        return
    from numba import typeof
    from numba.cuda.cudadrv.driver import binding, driver
    kern = _bls_shared.compile(tuple(typeof(a) for a in args))
    cufunc = kern._codelibrary.get_cufunc()
    attr = binding.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    driver.cuKernelSetAttribute(attr, int(nbytes), cufunc.handle, cufunc.device.id)
    _smem_allowed[dev] = nbytes


# shared-memory size classes: frequencies are launched in groups by nb so that the many
# short-period frequencies are not given the shared memory of the longest period
_NB_CLASSES = (1024, 2048, 3072, 4096, 6144, 8192, 12288)


def scan(freqs, nbs, time, flux, Rstar=1.0, Mstar=1.0, dips_only=False, global_mem_mb=512,
         max_shared_nb=None):
    """BLS over all frequencies (Period < baseline) on the GPU.

    Frequencies with nb <= max_nb() keep their bins in shared memory; the others (long
    periods with a bin-width cap) use global memory, in launches of at most
    ``global_mem_mb`` MB of bins (max_shared_nb lowers the shared-memory limit, for
    tests). Returns p (float64), jn1, jn2 (int32), the same quantities as bls_c.scan.
    """
    if not available():
        raise RuntimeError(f"CUDA BLS unavailable: {load_error()}")
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    nbs = np.ascontiguousarray(np.broadcast_to(nbs, freqs.shape), dtype=np.int64)
    time = np.ascontiguousarray(time, dtype=np.float64)
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    npt = time.size
    kmi, kma, kkmi = limits(freqs, nbs, npt, Rstar, Mstar)
    packed, cb, inv_scale = pack_flux(flux)
    smax = np.float32(0.0 if dips_only else np.inf)

    nf = freqs.size
    d = {name: cuda.to_device(a) for name, a in
         (("time", time), ("packed", packed), ("freqs", freqs), ("nbs", nbs),
          ("kmi", kmi), ("kma", kma), ("kkmi", kkmi))}
    p_d = cuda.device_array(nf, dtype=np.float64)
    j1_d = cuda.device_array(nf, dtype=np.int32)
    j2_d = cuda.device_array(nf, dtype=np.int32)
    common = (d["time"], d["packed"], d["freqs"], d["nbs"], d["kmi"], d["kma"], d["kkmi"])
    scal = (np.int64(npt), np.int64(cb), np.float64(inv_scale), smax)
    nbmax = max_nb() if max_shared_nb is None else min(int(max_shared_nb), max_nb())

    with warnings.catch_warnings():
        # the long-period groups are small launches; low occupancy there is expected
        warnings.filterwarnings("ignore", message="Grid size .* low occupancy")
        _launch(nbs, nbmax, common, scal, p_d, j1_d, j2_d, global_mem_mb)
    cuda.synchronize()
    return p_d.copy_to_host(), j1_d.copy_to_host(), j2_d.copy_to_host()


def _launch(nbs, nbmax, common, scal, p_d, j1_d, j2_d, global_mem_mb):
    lo = 0
    for hi in _NB_CLASSES + (nbmax,):
        hi = min(hi, nbmax)
        sel = np.flatnonzero((nbs > lo) & (nbs <= hi))
        lo = max(lo, hi)
        if sel.size == 0:
            continue
        args = common + (cuda.to_device(sel.astype(np.int32)),) + scal + (p_d, j1_d, j2_d)
        _allow_smem(nbmax * 8, args)
        _bls_shared[sel.size, THREADS, 0, int(nbs[sel].max()) * 8](*args)

    big = np.flatnonzero(nbs > nbmax)
    if big.size:
        nbl = int(nbs[big].max())
        rows = max(1, min(big.size, int(global_mem_mb * 2**20 // (16 * nbl))))
        gbins = cuda.device_array((rows, nbl), dtype=np.int64)
        gfc = cuda.device_array((rows, nbl), dtype=np.complex64)
        for c0 in range(0, big.size, rows):
            sel = big[c0:c0 + rows]
            args = common + (cuda.to_device(sel.astype(np.int32)),) + scal + (p_d, j1_d, j2_d, gbins, gfc)
            _bls_global[sel.size, THREADS](*args)
