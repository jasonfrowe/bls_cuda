"""Compiled (Numba) kernels for the polynomial detrender and the derivative outlier cut.

These reproduce transitPy5.polyfilter_reference (polyrange + polydetrend) and the
original cutoutliers point for point; only the mechanics differ:

- the light curve is sorted once and each window [t - boxbin/2, t + boxbin/2], clipped at
  data gaps, is found with a binary search instead of a mask over the whole array;
- each weighted polynomial fit is solved from its normal equations (order <= 3, abscissa
  centred on the point and scaled by the half-window) instead of np.polyfit;
- multipro = 1 runs the points on Numba threads instead of a process pool.

The trend values agree with the reference to ~1e-12 of the flux, except where a point lies
within rounding of the 4-sigma clipping boundary of a fit (rare; the trend then differs by
a fraction of the noise).
"""
import numpy as np
from numba import njit, prange

MAXITER = 10      # polydetrend: maximum clipping iterations
SIGCUT = 4.0      # polydetrend: clipping threshold (sigma)


@njit(cache=True)
def _solve(A, b):
    """Solve A x = b in place (Gaussian elimination, partial pivoting). False if singular."""
    n = b.shape[0]
    scale = 0.0
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > scale:
                scale = abs(A[i, j])
    if scale == 0.0:
        return False
    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[p, k]):
                p = i
        if abs(A[p, k]) <= 1e-13 * scale:
            return False
        if p != k:
            for j in range(n):
                tmp = A[k, j]; A[k, j] = A[p, j]; A[p, j] = tmp
            tmp = b[k]; b[k] = b[p]; b[p] = tmp
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= f * A[k, j]
            b[i] -= f * b[k]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i, j] * b[j]
        b[i] = s / A[i, i]
    return True


@njit(cache=True)
def _fit(x, y, w, use, npar, coef):
    """Weighted least-squares polynomial (npar coefficients, lowest order first) on use[k]."""
    S = np.zeros(2 * npar - 1)
    T = np.zeros(npar)
    nuse = 0
    for k in range(x.shape[0]):
        if use[k]:
            nuse += 1
            xp = w[k]
            for j in range(2 * npar - 1):
                S[j] += xp
                if j < npar:
                    T[j] += xp * y[k]
                xp *= x[k]
    if nuse < npar:
        return False
    A = np.empty((npar, npar))
    for i in range(npar):
        for j in range(npar):
            A[i, j] = S[i + j]
    if not _solve(A, T):
        return False
    for i in range(npar):
        coef[i] = T[i]
    return True


@njit(cache=True)
def _peval(coef, npar, x):
    v = 0.0
    for j in range(npar - 1, -1, -1):
        v = v * x + coef[j]
    return v


@njit(cache=True)
def _polydetrend(x, y, z, nfitp):
    """Trend at x = 0 from a polynomial fit with iterative 4-sigma clipping (= polydetrend).

    x is already centred on the point (and scaled); z are the flux errors (weights 1/z^2).
    """
    n = x.shape[0]
    npar = nfitp + 1
    w = 1.0 / (z * z)
    use = np.ones(n, dtype=np.bool_)
    coef = np.zeros(npar)
    if not _fit(x, y, w, use, npar, coef):
        return np.nan
    chisq = 0.0
    for k in range(n):
        r = (y[k] - _peval(coef, npar, x[k])) / z[k]
        chisq += r * r
    ochi = chisq
    it = 0
    dchi = 1.0
    res = np.empty(n)
    while it < MAXITER and dchi > 0.1:
        mean = 0.0
        for k in range(n):
            res[k] = y[k] - _peval(coef, npar, x[k])
            mean += res[k]
        mean /= n
        var = 0.0
        for k in range(n):
            var += (res[k] - mean) ** 2
        std = np.sqrt(var / n)
        for k in range(n):
            use[k] = abs(res[k]) < SIGCUT * std
        if not _fit(x, y, w, use, npar, coef):
            break
        chisq = 0.0
        for k in range(n):
            if use[k]:
                r = (y[k] - _peval(coef, npar, x[k])) / z[k]
                chisq += r * r
        dchi = abs(chisq - ochi)
        ochi = chisq
        it += 1
    return coef[0]


@njit(cache=True)
def _window(t, tzero, half, gaps):
    """Index range [lo, hi) of points with tzero - half < t < tzero + half, clipped at gaps."""
    t1 = tzero - half
    t2 = tzero + half
    if gaps.shape[0] > 0:
        j = np.searchsorted(gaps, tzero)            # gaps[j-1] < tzero <= gaps[j]
        if j > 0 and gaps[j - 1] > t1:
            t1 = gaps[j - 1]
        k = np.searchsorted(gaps, tzero, side="right")
        if k < gaps.shape[0] and gaps[k] < t2:
            t2 = gaps[k]
    lo = np.searchsorted(t, t1, side="right")
    hi = np.searchsorted(t, t2, side="left")
    return lo, hi


@njit(cache=True)
def _gather(t, f, e, msk, lo, hi, tzero, scale, unmasked_only):
    n = 0
    for k in range(lo, hi):
        if (not unmasked_only) or msk[k] == 0:
            n += 1
    x = np.empty(n); y = np.empty(n); z = np.empty(n)
    m = 0
    for k in range(lo, hi):
        if (not unmasked_only) or msk[k] == 0:
            x[m] = (t[k] - tzero) / scale
            y[m] = f[k]
            z[m] = e[k]
            m += 1
    return x, y, z


@njit(cache=True)
def _trend_one(i, t, f, e, msk, gaps, bbd2, nfitp):
    """Trend at sorted index i (= transitPy5.polyrange)."""
    tzero = t[i]
    lo1, hi1 = _window(t, tzero, bbd2, gaps)
    x, y, z = _gather(t, f, e, msk, lo1, hi1, tzero, bbd2, True)
    npt2 = x.shape[0]

    if msk[i] == 0:
        if npt2 > nfitp + 1:
            v = _polydetrend(x, y, z, nfitp)
            if np.isfinite(v):
                return v
        if npt2 > 0:
            return np.mean(y)
        return f[i]

    # a masked point: the trend must come from unmasked points on both sides
    min_pts = 2 * (nfitp + 1)
    both = False
    for scale in (1.0, 2.0, 4.0):
        if scale > 1.0:
            lo, hi = _window(t, tzero, bbd2 * scale, gaps)
            x, y, z = _gather(t, f, e, msk, lo, hi, tzero, bbd2, True)
        npt2 = x.shape[0]
        left = False
        right = False
        for k in range(npt2):
            if x[k] < 0.0:
                left = True
            elif x[k] > 0.0:
                right = True
        both = left and right
        if npt2 >= min_pts and both:
            v = _polydetrend(x, y, z, nfitp)
            if np.isfinite(v):
                return v

    # fallbacks, on the widest window
    if npt2 == 0:
        if hi1 > lo1:
            return np.median(f[lo1:hi1])
        return f[i]
    if both and npt2 >= 3:
        v = _polydetrend(x, y, z, min(nfitp, 1))
        if np.isfinite(v):
            return v
    order = np.argsort(np.abs(x))
    nn = min(min_pts, npt2)
    near = np.empty(nn)
    for k in range(nn):
        near[k] = y[order[k]]
    return np.median(near)


@njit(cache=True)
def _trend_serial(t, f, e, msk, gaps, bbd2, nfitp):
    n = t.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = _trend_one(i, t, f, e, msk, gaps, bbd2, nfitp)
    return out


@njit(cache=True, parallel=True)
def _trend_parallel(t, f, e, msk, gaps, bbd2, nfitp):
    n = t.shape[0]
    out = np.empty(n)
    for i in prange(n):
        out[i] = _trend_one(i, t, f, e, msk, gaps, bbd2, nfitp)
    return out


def polyfilter_trend(time, flux, ferr, tflag, boxbin, nfitp, gapsize, multipro=0):
    """Trend (offset) at every point, in the input order. See transitPy5.polyfilter."""
    time = np.asarray(time, dtype=np.float64)
    ts = np.argsort(time, kind="stable")
    t = np.ascontiguousarray(time[ts])
    f = np.ascontiguousarray(np.asarray(flux, dtype=np.float64)[ts])
    e = np.ascontiguousarray(np.asarray(ferr, dtype=np.float64)[ts])
    m = np.ascontiguousarray((np.asarray(tflag) != 0).astype(np.int8)[ts])
    dt = np.diff(t)
    big = dt > gapsize
    gaps = np.ascontiguousarray(0.5 * (t[1:][big] + t[:-1][big]))
    kern = _trend_parallel if multipro else _trend_serial
    tr = kern(t, f, e, m, gaps, boxbin / 2.0, int(nfitp))
    out = np.empty_like(tr)
    out[ts] = tr
    return out


@njit(cache=True)
def _cutoutliers(y, nsampmax, sigma, up_only):
    npt = y.shape[0]
    icut = np.zeros(npt, dtype=np.int32)
    threshold = 0.0005
    d = np.empty(2 * nsampmax)
    for i in range(1, npt - 1):
        i1 = max(0, i - nsampmax)
        i2 = min(npt - 1, i + nsampmax)
        nd = i2 - i1
        for k in range(nd):
            d[k] = abs(y[i1 + k + 1] - y[i1 + k])
        std = np.median(d[:nd])
        if sigma > 0:
            threshold = std * sigma
        vp = y[i] - y[i + 1]
        vm = y[i] - y[i - 1]
        if up_only:
            if vp > threshold and vm > threshold:
                icut[i] = 1
        elif abs(vp) > threshold and abs(vm) > threshold and vp / vm > 0:
            icut[i] = 1
    return icut


def cutoutliers(x, y, nsampmax=3, sigma=3.0, direction="both"):
    """Derivative-based outlier cut (see transitPy5.cutoutliers). Data in time order."""
    return _cutoutliers(np.ascontiguousarray(y, dtype=np.float64), int(nsampmax), float(sigma),
                        direction == "up")
