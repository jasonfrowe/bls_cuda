"""Complete elliptic integrals for the Mandel & Agol (2002) quadratic model.

All three use Bulirsch's general complete elliptic integral cel(kc, p, a, b), as in
occultquad.f of TransitFit5 (which replaced Carlson's algorithm in Aug 2026). The
earlier Python versions used Hastings polynomial approximations for K and E, which
differ from the exact values by up to ~1e-8 and changed transit depths by ~0.01 ppm.
"""
import numpy as np
from numba import njit

_CA = 1e-12  # relative convergence of the AGM iteration (as in occultquad.f)


@njit(cache=True)
def cel(kc, p, a, b):
    """
    Bulirsch's general complete elliptic integral,
    cel(kc, p, a, b) = int_0^{pi/2} (a cos^2 t + b sin^2 t) / ((cos^2 t + p sin^2 t) sqrt(cos^2 t + kc^2 sin^2 t)) dt.
    Returns 0 for kc = 0, as in occultquad.f.
    """
    if kc == 0.0:
        return 0.0

    kc = abs(kc)
    e = kc
    m = 1.0

    if p > 0.0:
        p = np.sqrt(p)
        b = b / p
    else:
        f = kc * kc
        g = 1.0 - p
        e = (f - p) / g
        h = (b - a * p) / g
        p = np.sqrt(e / g)
        a = (a - b) / g
        b = -h / (p * g) + a * p

    while True:
        f = a
        a = a + b / p
        g = e / p
        b = b + f * g
        b = b + b
        p = g + p
        g = m
        m = m + kc
        if abs(g - kc) <= g * _CA:
            break
        kc = 2.0 * np.sqrt(e)
        e = kc * m

    return 0.5 * np.pi * (a * m + b) / (m * (m + p))


@njit(cache=True)
def ellK(k):
    """Complete elliptic integral of the first kind, K(k), for modulus k."""
    return cel(np.sqrt(max(0.0, 1.0 - k*k)), 1.0, 1.0, 1.0)


@njit(cache=True)
def ellE(k):
    """Complete elliptic integral of the second kind, E(k), for modulus k."""
    kc = np.sqrt(max(0.0, 1.0 - k*k))
    return cel(kc, 1.0, 1.0, kc*kc)


@njit(cache=True)
def ellPi(n, k):
    """
    Complete elliptic integral of the third kind, Pi(n, k) = int dt / ((1 - n sin^2 t) sqrt(1 - k^2 sin^2 t)).
    (occultquad.f writes the characteristic with the opposite sign: cel(kc, 1 + n_F, 1, 1), n_F = -n.)
    """
    return cel(np.sqrt(max(0.0, 1.0 - k*k)), 1.0 - n, 1.0, 1.0)
