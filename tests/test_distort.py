import numpy as np
import pytfit5.transitPy5 as tpy5
from pytfit5 import distort as ds

CAD = 29.4244 / 1440


def test_weights_reproduce_polynomials():
    h = ds.filter_weights(2.0, 3, CAD)
    x = np.arange(-(h.size // 2), h.size // 2 + 1) * CAD
    for c in ([1, 0, 0, 0], [0.3, 2.0, -1.0, 0.5]):
        assert np.isclose(np.sum(h * np.polyval(c[::-1], x)), c[0])


def test_distorted_box_matches_polyfilter():
    t = np.arange(0, 30, CAD)
    rng = np.random.default_rng(2)
    for W, D in ((2.0, 0.25), (5.0, 0.5)):
        s = ds.box(t - 15, D, CAD)
        f = 1 + 1e-3 * s + 1e-5 * rng.standard_normal(t.size)
        out = tpy5.polyfilter(t, f, np.full(t.size, 1e-5), np.zeros(t.size), W, 3, 1.0, multipro=0)
        d = (out - np.median(out)) / 1e-3
        g = ds.template(t, 1e4, 15.0, D, W, 3, CAD, "distorted")
        near = np.abs(t - 15) < W
        assert np.std((d - g)[near]) < 0.03
