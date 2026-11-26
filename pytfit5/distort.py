"""Transit templates distorted by the polynomial detrender (polyfilter).

For data with even sampling and no gaps, and signals too small to trigger its sigma clipping,
polyfilter is a linear, time-invariant filter: the trend at each point is the value at that
point of a least-squares polynomial (order nfitp) fitted over +/- boxbin/2, i.e. a weighted sum
of the neighbouring fluxes with Savitzky-Golay weights h. A transit s therefore comes out of the
detrender as g = s - h * s: shallower, with shoulders reaching about boxbin/2 beyond each edge.
Correlating the detrended light curve with g (a matched filter) instead of with a box recovers
the signal in the shoulders, as the Kepler TPS does with its whitened templates (Jenkins 2002).
"""
import numpy as np


def filter_weights(boxbin, nfitp, cadence):
    """Weights h_k (k = -m..m) such that the polyfilter trend at a point is sum_k h_k f_{i+k}.

    boxbin: full window width (days); nfitp: polynomial order; cadence: sampling step (days).
    """
    m = int(np.floor(0.5 * boxbin / cadence + 1e-9))
    x = np.arange(-m, m + 1) * cadence / (0.5 * boxbin)
    v = np.vander(x, nfitp + 1, increasing=True)
    return np.linalg.pinv(v)[0]          # value of the fitted polynomial at x = 0


def box(tau, dur, cadence=0.0):
    """Unit box transit (-1 in transit) at offsets tau from mid-transit, integrated over an
    exposure of length cadence (0: instantaneous)."""
    tau = np.abs(np.asarray(tau, float))
    if cadence <= 0:
        return -(tau < 0.5 * dur).astype(float)
    lo, hi = tau - 0.5 * cadence, tau + 0.5 * cadence
    ov = np.clip(np.minimum(hi, 0.5 * dur) - np.maximum(lo, -0.5 * dur), 0, None)
    return -ov / cadence


def distorted_box(dur, boxbin, nfitp=3, cadence=29.4244 / 1440, osamp=4):
    """Response of the detrender to a unit box transit of duration dur.

    Returns (tau, g): offsets from mid-transit (days, step cadence/osamp) and the detrended
    signal, which is zero beyond |tau| > dur/2 + boxbin/2.
    """
    h = filter_weights(boxbin, nfitp, cadence)
    m = (h.size - 1) // 2
    step = cadence / osamp
    half = 0.5 * dur + (m + 1) * cadence
    tau = np.arange(-np.ceil(half / step), np.ceil(half / step) + 1) * step
    s = box(tau, dur, cadence)
    trend = np.zeros_like(tau)
    for k in range(-m, m + 1):           # trend(tau) = sum_k h_k s(tau + k cadence)
        trend += h[k + m] * box(tau + k * cadence, dur, cadence)
    return tau, s - trend


def template(t, per, t0, dur, boxbin, nfitp=3, cadence=29.4244 / 1440, kind="distorted"):
    """Template at times t for a transit train (per, t0, dur): kind "box" or "distorted".

    For the distorted template the response of neighbouring transits is summed, which matters
    when per < dur + boxbin.
    """
    t = np.asarray(t, float)
    if kind == "box":
        ph = (t - t0 + 0.5 * per) % per - 0.5 * per
        return box(ph, dur, cadence)
    tau, g = distorted_box(dur, boxbin, nfitp, cadence)
    out = np.zeros_like(t)
    ph = (t - t0 + 0.5 * per) % per - 0.5 * per
    kmax = int(np.ceil(tau[-1] / per)) + 1
    for k in range(-kmax, kmax + 1):
        out += np.interp(ph + k * per, tau, g, left=0.0, right=0.0)
    return out
