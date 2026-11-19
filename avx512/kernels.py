"""Experimental BLS kernel variants (Numba) for the AVX-512 investigation.

All take the same arguments as bls_cpu.bls_kernel and return (p, jn1, jn2).
"""
import numpy as np
from numba import njit
from pytfit5.bls_cpu import pifac, Rsun, G, Msun, day2sec, onehour


@njit(cache=True, inline="always")
def _limits(freq, nb, npt, Rstar, Mstar):
    fsec = freq / day2sec
    q = pifac * Rstar * Rsun / (G * Mstar * Msun) ** (1.0 / 3.0) * fsec ** (2.0 / 3.0)
    qmi = q / 2.0
    if qmi < onehour * freq / 2:
        qmi = onehour * freq / 2
    qma = q * 2.0
    if qma > 0.25:
        qma = 0.25
    if qmi > qma:
        qma = qmi * 2.0
        if qma > 0.4:
            qma = 0.4
    kmi = int(qmi * nb)
    if kmi < 1:
        kmi = 1
    kma = int(qma * nb) + 1
    kkmi = int(npt * qmi)
    if kkmi < 5:
        kkmi = 5
    return kmi, kma, kkmi


@njit(cache=True, error_model="numpy")
def _bin(freq, time, flux, nb):
    npt = time.shape[0]
    ibi = np.zeros(2 * nb)
    y = np.zeros(2 * nb)
    jbin = np.empty(npt, dtype=np.int32)
    fnbm1 = float(nb - 1)
    for i in range(npt):
        tf = time[i] * freq
        phase = tf - np.int32(tf)
        jbin[i] = np.int32(phase * fnbm1)
    for i in range(npt):
        j = jbin[i]
        ibi[j] += 1.0
        y[j] += flux[i]
    for i in range(nb):
        ibi[i + nb] = ibi[i]
        y[i + nb] = y[i]
    return ibi, y


@njit(cache=True, error_model="numpy")
def _search_div(ibi, y, nb, npt, kmi, kma, kkmi):
    """Current bls_kernel search (division in inner loop)."""
    fnpt = float(npt)
    fkkmi = float(kkmi)
    S = np.zeros(nb)
    K = np.zeros(nb)
    B = np.zeros(nb)
    E = np.zeros(nb, dtype=np.int64)
    for m in range(kma + 1):
        if m + 1 <= kmi:
            for i in range(nb):
                S[i] += y[i + m]
                K[i] += ibi[i + m]
        else:
            for i in range(nb):
                s = S[i] + y[i + m]
                kk = K[i] + ibi[i + m]
                S[i] = s
                K[i] = kk
                dfac = kk * (fnpt - kk)
                pow1 = s * s / dfac
                better = (kk > fkkmi) & (dfac > 0.0) & (pow1 > B[i])
                B[i] = pow1 if better else B[i]
                E[i] = i + m if better else E[i]
    p = 0.0
    jn1 = 0
    jn2 = 0
    for i in range(nb):
        if B[i] > p:
            p = B[i]
            jn1 = i
            jn2 = E[i]
    return p, jn1, jn2


@njit(cache=True, error_model="numpy")
def _search_nodiv(ibi, y, nb, npt, kmi, kma, kkmi):
    """Division-free search: keep best (s^2, dfac) per start bin and compare
    s^2 * Dbest > Nbest * dfac (both dfac > 0), so no divide in the hot loop."""
    fnpt = float(npt)
    fkkmi = float(kkmi)
    S = np.zeros(nb)
    K = np.zeros(nb)
    N = np.zeros(nb)          # best s^2
    D = np.ones(nb)           # best dfac (N/D = 0 initially)
    E = np.zeros(nb, dtype=np.int64)
    for m in range(kma + 1):
        if m + 1 <= kmi:
            for i in range(nb):
                S[i] += y[i + m]
                K[i] += ibi[i + m]
        else:
            for i in range(nb):
                s = S[i] + y[i + m]
                kk = K[i] + ibi[i + m]
                S[i] = s
                K[i] = kk
                dfac = kk * (fnpt - kk)
                s2 = s * s
                better = (kk > fkkmi) & (dfac > 0.0) & (s2 * D[i] > N[i] * dfac)
                N[i] = s2 if better else N[i]
                D[i] = dfac if better else D[i]
                E[i] = i + m if better else E[i]
    p = 0.0
    jn1 = 0
    jn2 = 0
    for i in range(nb):
        b = N[i] / D[i]
        if b > p:
            p = b
            jn1 = i
            jn2 = E[i]
    return p, jn1, jn2


@njit(cache=True, error_model="numpy")
def bin_only(freq, time, flux, nb, Rstar, Mstar):
    nb = int(nb)
    ibi, y = _bin(freq, time, flux, nb)
    return y[0], 0, 0


@njit(cache=True, error_model="numpy")
def search_only_div(freq, time, flux, nb, Rstar, Mstar):
    """Search on fixed (frequency-independent) bins: isolates search cost."""
    nb = int(nb)
    npt = time.shape[0]
    ibi = np.empty(2 * nb)
    y = np.empty(2 * nb)
    for i in range(2 * nb):
        ibi[i] = npt / nb
        y[i] = np.sin(i * 0.01) * 1e-4
    kmi, kma, kkmi = _limits(freq, nb, npt, Rstar, Mstar)
    return _search_div(ibi, y, nb, npt, kmi, kma, kkmi)


@njit(cache=True, error_model="numpy")
def search_only_nodiv(freq, time, flux, nb, Rstar, Mstar):
    nb = int(nb)
    npt = time.shape[0]
    ibi = np.empty(2 * nb)
    y = np.empty(2 * nb)
    for i in range(2 * nb):
        ibi[i] = npt / nb
        y[i] = np.sin(i * 0.01) * 1e-4
    kmi, kma, kkmi = _limits(freq, nb, npt, Rstar, Mstar)
    return _search_nodiv(ibi, y, nb, npt, kmi, kma, kkmi)


@njit(cache=True, error_model="numpy")
def bls_kernel_nodiv(freq, time, flux, nb, Rstar, Mstar):
    nb = int(nb)
    npt = time.shape[0]
    ibi, y = _bin(freq, time, flux, nb)
    kmi, kma, kkmi = _limits(freq, nb, npt, Rstar, Mstar)
    return _search_nodiv(ibi, y, nb, npt, kmi, kma, kkmi)


@njit(cache=True, error_model="numpy")
def _search_tiled(ibi, y, nb, npt, kmi, kma, kkmi, T):
    """Same arithmetic as _search_div, but start bins are processed in tiles of T
    so the per-bin state (S, K, B, E) stays in L1 across all widths m."""
    fnpt = float(npt)
    fkkmi = float(kkmi)
    S = np.zeros(T)
    K = np.zeros(T)
    B = np.zeros(T)
    E = np.zeros(T, dtype=np.int64)
    p = 0.0
    jn1 = 0
    jn2 = 0
    for i0 in range(0, nb, T):
        n = min(T, nb - i0)
        for t in range(n):
            S[t] = 0.0
            K[t] = 0.0
            B[t] = 0.0
            E[t] = 0
        for m in range(kma + 1):
            yo = y[i0 + m: i0 + m + n]
            io = ibi[i0 + m: i0 + m + n]
            if m + 1 <= kmi:
                for t in range(n):
                    S[t] += yo[t]
                    K[t] += io[t]
            else:
                for t in range(n):
                    s = S[t] + yo[t]
                    kk = K[t] + io[t]
                    S[t] = s
                    K[t] = kk
                    dfac = kk * (fnpt - kk)
                    pow1 = s * s / dfac
                    better = (kk > fkkmi) & (dfac > 0.0) & (pow1 > B[t])
                    B[t] = pow1 if better else B[t]
                    E[t] = i0 + t + m if better else E[t]
        for t in range(n):
            if B[t] > p:
                p = B[t]
                jn1 = i0 + t
                jn2 = E[t]
    return p, jn1, jn2


def _make_tiled(T):
    @njit(cache=False, error_model="numpy")
    def k(freq, time, flux, nb, Rstar, Mstar):
        nb = int(nb)
        npt = time.shape[0]
        ibi, y = _bin(freq, time, flux, nb)
        kmi, kma, kkmi = _limits(freq, nb, npt, Rstar, Mstar)
        return _search_tiled(ibi, y, nb, npt, kmi, kma, kkmi, T)
    return k


tiled32 = _make_tiled(32)
tiled64 = _make_tiled(64)
tiled128 = _make_tiled(128)
tiled256 = _make_tiled(256)
tiled512 = _make_tiled(512)


@njit(cache=True, error_model="numpy")
def _search_tiled_nodiv(ibi, y, nb, npt, kmi, kma, kkmi, T):
    """Tiled + division-free: best kept as (N=s^2, D=dfac); compare s^2*D > N*dfac."""
    fnpt = float(npt)
    fkkmi = float(kkmi)
    S = np.zeros(T)
    K = np.zeros(T)
    N = np.zeros(T)
    D = np.ones(T)
    E = np.zeros(T, dtype=np.int64)
    p = 0.0
    jn1 = 0
    jn2 = 0
    for i0 in range(0, nb, T):
        n = min(T, nb - i0)
        for t in range(n):
            S[t] = 0.0
            K[t] = 0.0
            N[t] = 0.0
            D[t] = 1.0
            E[t] = 0
        for m in range(kma + 1):
            yo = y[i0 + m: i0 + m + n]
            io = ibi[i0 + m: i0 + m + n]
            if m + 1 <= kmi:
                for t in range(n):
                    S[t] += yo[t]
                    K[t] += io[t]
            else:
                for t in range(n):
                    s = S[t] + yo[t]
                    kk = K[t] + io[t]
                    S[t] = s
                    K[t] = kk
                    dfac = kk * (fnpt - kk)
                    s2 = s * s
                    better = (kk > fkkmi) & (dfac > 0.0) & (s2 * D[t] > N[t] * dfac)
                    N[t] = s2 if better else N[t]
                    D[t] = dfac if better else D[t]
                    E[t] = i0 + t + m if better else E[t]
        for t in range(n):
            b = N[t] / D[t]
            if b > p:
                p = b
                jn1 = i0 + t
                jn2 = E[t]
    return p, jn1, jn2


def _make_tiled_nodiv(T):
    @njit(cache=False, error_model="numpy")
    def k(freq, time, flux, nb, Rstar, Mstar):
        nb = int(nb)
        npt = time.shape[0]
        ibi, y = _bin(freq, time, flux, nb)
        kmi, kma, kkmi = _limits(freq, nb, npt, Rstar, Mstar)
        return _search_tiled_nodiv(ibi, y, nb, npt, kmi, kma, kkmi, T)
    return k


tnd64 = _make_tiled_nodiv(64)
tnd128 = _make_tiled_nodiv(128)
tnd256 = _make_tiled_nodiv(256)
tnd512 = _make_tiled_nodiv(512)
