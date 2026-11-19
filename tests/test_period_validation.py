"""Period validation: aliases, single transits, and the bls_pulse single-event flag."""
import numpy as np
import pytest

from pytfit5 import period_validation as pval
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.keplerian as kep


def lc(per, t0, snr, seed, time_length=60.0, noise="none"):
    phot, sol = generate_synthetic_lightcurve(t0=t0, per=per, time_length=time_length, depth=0.002,
                                              snr=snr, cadence=1 / 96, stellar_noise_type=noise,
                                              stellar_noise_amplitude=0.0005, seed=seed)
    return phot.time, phot.flux - np.median(phot.flux), kep.transitDuration(sol)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("given, want", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (1 / 3, 3.0)])
def test_aliases_recover_true_period(seed, given, want):
    rng = np.random.default_rng(seed)
    P, t0 = rng.uniform(2.0, 8.0), rng.uniform(0.0, 2.0)
    time, flux, tdur = lc(P, t0, snr=15.0, seed=seed)
    r = pval.test_period_aliases(time, flux, t0, P * given, tdur)
    assert r["best_factor"] == pytest.approx(want)
    assert r["best_period"] == pytest.approx(P, rel=1e-6)
    assert r["is_alias"] == (want != 1.0)


def test_bls_epoch_is_tested():
    """The epoch grid includes t0_initial, so the BLS epoch is never skipped."""
    time, flux, tdur = lc(6.2, 3.0, snr=15.0, seed=7)
    t0, snr, _, _ = pval.optimize_epoch(time, flux, 6.2, tdur, 3.0, search_window=2 * tdur)
    snr_exact = pval.compute_transit_snr(time, flux, 3.0, 6.2, tdur)[0]
    assert snr >= snr_exact


def test_individual_transit_depths_are_relative_to_baseline():
    time, flux, tdur = lc(6.2, 3.0, snr=20.0, seed=4)
    ti = pval.analyze_individual_transits(time, flux, 3.0, 6.2, tdur)
    assert np.all(np.abs(ti["transit_depths"] - 0.002) < 0.0015)
    assert not ti["is_single_dominant"]
    assert ti["rest_snr"] > ti["strongest_snr"]


def test_single_transit_recognised():
    time, flux, tdur = lc(85.0, 20.0, snr=15.0, seed=5)
    fake = type("Ans", (), dict(epo=20.0, bper=12.0, tdur=tdur))()
    res, ti = pval.validate_bls_period(time, flux, fake, verbose=False)
    assert res["is_single_transit"] or ti["is_single_dominant"]


def test_bls_pulse_sets_single_flag():
    import pytfit5.bls_cpu as gbls
    import pytfit5.transitPy5 as tpy5
    inp = tpy5.tpy5_inputs_class()
    inp.plots = 0
    phot, _ = generate_synthetic_lightcurve(t0=10.0, per=85.0, time_length=60.0, depth=0.008, snr=12.0,
                                            cadence=1 / 48, stellar_noise_type="none", seed=11)
    ans = gbls.bls_pulse(inp, phot.time, phot.flux)
    assert ans.single and ans.bper > np.ptp(phot.time)
    assert abs(ans.epo - 10.0) < 0.1
    phot, _ = generate_synthetic_lightcurve(t0=1.0, per=4.0, time_length=40.0, depth=0.002, snr=20.0,
                                            cadence=1 / 48, stellar_noise_type="none", seed=12)
    ans = gbls.bls_pulse(inp, phot.time, phot.flux)
    assert not ans.single and ans.bper == pytest.approx(4.0, rel=1e-3)
