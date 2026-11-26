"""generate_synthetic_lightcurve: exposure time (itime) is independent of the cadence, and nintg is
passed through to the model; the defaults (itime = cadence, nintg = 41) are unchanged."""
import numpy as np

import pytfit5.transitmodel as tm
from pytfit5.synthetic import generate_synthetic_lightcurve

KW = dict(t0=2.71, per=5.37, time_length=20.0, depth=0.03 ** 2, snr=30.0, cadence=12.1 / 1440,
          stellar_noise_type="none", seed=3)


def _clean(phot, sol, nintg):
    return tm.transitModel(sol.to_array(), phot.time, phot.itime, nintg=nintg)


def test_default_itime_is_cadence():
    phot, _ = generate_synthetic_lightcurve(**KW)
    assert np.all(phot.itime == KW["cadence"])


def test_itime_decoupled_from_cadence():
    kw = dict(KW, snr=1e12)    # noise ~1e-15: the flux is the model
    phot, sol = generate_synthetic_lightcurve(itime=67 / 86400, nintg=11, **kw)
    assert np.all(phot.itime == 67 / 86400)
    assert np.allclose(np.diff(phot.time), KW["cadence"])
    model = _clean(phot, sol, 11)
    assert np.max(np.abs(phot.flux - model / np.median(model))) < 1e-12
    # a 67-s exposure is not smeared like the 12.1-min one
    ref, _ = generate_synthetic_lightcurve(**kw)
    assert np.max(np.abs(phot.flux - ref.flux)) > 1e-5
