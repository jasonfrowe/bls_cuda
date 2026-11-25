"""The Python transit model must match the Fortran TransitFit5, and its orbit convention must
match the literature.

tests/data/transitmodel_fortran_ref.npz holds light curves computed with transitmodel.f
(github.com/jasonfrowe/Kepler, branch omega-convention, gfortran -O2) for 20 parameter sets:
quadratic, Kipping and non-linear limb darkening, b = 0, b < 0, grazing, dilution,
eccentric orbits at four values of omega, occultations, ellipsoidal, albedo and Doppler
terms, two planets, Rp/R* < 0 and TTVs (one and five measurements).

omega is the argument of periastron of the radial-velocity literature (Winn 2010): the
transit is at true anomaly f = pi/2 - omega, the occultation at f = 3pi/2 - omega, and the
star's radial velocity is K [cos(f + omega) + e cos(omega)].
"""
from pathlib import Path

import numpy as np
import pytest

from pytfit5 import keplerian as kep
from pytfit5 import transitmcmc
from pytfit5.transitmodel import transitModel

REF = np.load(Path(__file__).parent / "data" / "transitmodel_fortran_ref.npz")
NAMES = list(REF["names"])


@pytest.mark.parametrize("k", range(len(NAMES)), ids=NAMES)
def test_matches_fortran(k):
    sol, ttv = REF[f"sol{k}"], REF[f"ttv{k}"]
    kw = {}
    if len(ttv):
        kw = dict(ntt=np.array([len(ttv)]), tobs=ttv[None, :, 0].copy(), omc=ttv[None, :, 1].copy())
    model = transitModel(sol, REF["time"], itime=REF["itime"], **kw)
    np.testing.assert_allclose(model, REF[f"model{k}"], rtol=0, atol=1e-13)  # 1e-7 ppm


def test_mcmc_prior_rejects_unbound_orbit():
    sol = REF["sol0"].copy()
    lb, ub = transitmcmc.setup_priors(sol)
    assert np.isfinite(transitmcmc.logprior(sol, lb, ub))
    sol[12], sol[13] = 0.8, 0.8          # ecw^2 + esw^2 = 1.28: both inside their own bounds
    assert transitmcmc.logprior(sol, lb, ub) == -np.inf


def test_mcmc_prior_quadratic_limb_darkening():
    sol = REF["sol1"].copy()             # u1 = 0.4, u2 = 0.25
    lb, ub = transitmcmc.setup_priors(sol)
    assert np.isfinite(transitmcmc.logprior(sol, lb, ub))
    sol[1], sol[2] = 0.9, 0.3            # u1 + u2 > 1: negative intensity at the limb
    assert transitmcmc.logprior(sol, lb, ub) == -np.inf


def _sol(e, omega_deg, b=0.0, rdr=0.1, ted=0.0, krv=0.0, per=10.0, t0=1.0):
    ecw, esw = kep.ecw_esw(e, np.radians(omega_deg))
    return np.array([1.4, 0, 0, 0.36, 0.30, 0, 0, 0, t0, per, b, rdr, ecw, esw, krv, ted, 0, 0], float)


def _time_of_true_anomaly(f, e, omega, t0, per):
    """Time at which the planet reaches true anomaly f, for mid-transit at t0."""
    def mean_anom(f):
        E = 2*np.arctan(np.sqrt((1 - e)/(1 + e))*np.tan(f/2))
        return E - e*np.sin(E)
    return t0 + per*np.mod(mean_anom(f) - mean_anom(np.pi/2 - omega), 2*np.pi)/(2*np.pi)


@pytest.mark.parametrize("e, omega_deg", [(0.3, 60.0), (0.3, 200.0), (0.5, 300.0), (0.1, 0.0)])
def test_occultation_at_superior_conjunction(e, omega_deg):
    """The occultation is centred on f = 3pi/2 - omega (for b = 0, the minimum sky separation)."""
    sol = _sol(e, omega_deg, ted=500.0)
    t_occ = _time_of_true_anomaly(3*np.pi/2 - np.radians(omega_deg), e, np.radians(omega_deg), 1.0, 10.0)
    t = np.linspace(t_occ - 0.5, t_occ + 0.5, 20001)
    flux = transitModel(sol, t, itime=np.full(t.size, 1e-4), nintg=1)
    dip = flux < flux.max() - 1e-7
    t_mid = 0.5*(t[dip][0] + t[dip][-1])
    assert abs(t_mid - t_occ) < 2e-4                       # 17 s
    # first-order timing offset (Winn 2010, eq. 33): e cos(omega) sets the occultation phase
    approx = 1.0 + 5.0*(1 + 4/np.pi*e*np.cos(np.radians(omega_deg)))
    assert abs(np.mod(t_occ - approx + 5, 10) - 5) < 10*e*e   # agrees to O(e^2)


def test_transit_duration_periastron_vs_apastron():
    """omega = 90 deg puts periastron at transit: shorter transit by (1 - e)/(1 + e) (for b = 0)."""
    peri, apo = (kep.transitDuration(_obj(_sol(0.4, w)), 0) for w in (90.0, 270.0))
    assert peri/apo == pytest.approx((1 - 0.4)/(1 + 0.4), rel=2e-3)


def _obj(sol):
    from pytfit5.transitmodel import transit_model_class
    o = transit_model_class(); o.from_array(sol)
    return o


@pytest.mark.parametrize("e, omega_deg", [(0.0, 0.0), (0.3, 60.0), (0.6, 250.0)])
def test_doppler_term_is_the_literature_radial_velocity(e, omega_deg):
    """The Doppler-boosting term of the photometric model, -v/c, uses v = K[cos(f+omega) + e cos(omega)]."""
    K = 3.0e4                                              # m/s, large enough to see the term
    sol = _sol(e, omega_deg, rdr=0.0, krv=K)
    t = np.linspace(0, 10, 501)
    flux = transitModel(sol, t, itime=np.full(t.size, 1e-6), nintg=1)
    v_phot = (1 - flux)*2.99792458e8
    v_rv = kep.radial_velocity(t, 1.0, 10.0, K, sol[12], sol[13])
    np.testing.assert_allclose(v_phot, v_rv, rtol=0, atol=1e-6*K)


def test_radial_velocity_sign_at_transit():
    """Circular orbit: the star moves away from us before transit and towards us after it."""
    v = kep.radial_velocity([0.99, 1.0, 1.01], 1.0, 10.0, 50.0, 0.0, 0.0)
    assert v[0] > 0 and abs(v[1]) < 1e-9 and v[2] < 0


@pytest.mark.parametrize("omega_deg, inc_deg, Omega_deg", [(60, 88, 0), (200, 89.5, 120), (300, 85, 250)])
def test_state_vectors_give_omega(omega_deg, inc_deg, Omega_deg):
    """Orbit built in the sky frame of Winn (2010) (Z towards the observer) and rotated by an
    arbitrary node angle; ecc_params_from_state recovers omega."""
    e, mu = 0.35, 1.0
    w, inc, Om = np.radians([omega_deg, inc_deg, Omega_deg])
    f = 1.234
    r = (1 - e*e)/(1 + e*np.cos(f))
    pos = r*np.array([-np.cos(w + f), -np.sin(w + f)*np.cos(inc), np.sin(w + f)*np.sin(inc)])
    # velocity from the derivative with respect to f (dr/df and d(w+f)/df), scaled by df/dt
    drdf = r*e*np.sin(f)/(1 + e*np.cos(f))
    dpos = drdf*pos/r + r*np.array([np.sin(w + f), -np.cos(w + f)*np.cos(inc), np.cos(w + f)*np.sin(inc)])
    dfdt = np.sqrt(mu*(1 - e*e))/(1 - e*e)**2*(1 + e*np.cos(f))**2
    vel = dpos*dfdt
    rot = np.array([[np.cos(Om), -np.sin(Om), 0], [np.sin(Om), np.cos(Om), 0], [0, 0, 1]])   # about the line of sight
    ecw, esw, e_out, w_out = kep.ecc_params_from_state(rot @ pos, rot @ vel, mu, observer=(0, 0, 1))
    assert e_out == pytest.approx(e, abs=1e-12)
    assert np.mod(w_out - w + np.pi, 2*np.pi) - np.pi == pytest.approx(0, abs=1e-12)
    assert (ecw, esw) == pytest.approx(kep.ecw_esw(e, w), abs=1e-12)


def test_legacy_conversion():
    """Old TransitFit5 (w = true anomaly at transit) to current parameters: a swap."""
    e, w_old = 0.2, np.radians(35.0)
    old = (np.sqrt(e)*np.cos(w_old), np.sqrt(e)*np.sin(w_old))
    new = kep.from_legacy_transitfit5(*old)
    assert kep.ecc_omega(*new)[1] == pytest.approx(np.pi/2 - w_old)
