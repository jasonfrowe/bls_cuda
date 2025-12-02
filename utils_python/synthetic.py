import numpy as np

from typing import Tuple, Optional

import pytfit5.transitmodel as transitm
from pytfit5.transitmodel import transitModel
import pytfit5.keplerian as kep


def generate_synthetic_lightcurve(
    t0: float,
    per: float,
    time_length: float,
    depth: float,
    snr: float,
    cadence: float = 1.0/48.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic lightcurve using PyTFit5 transit model for BLS testing.

    Parameters
    - t0: Transit center time (days)
    - per: Orbital period (days)
    - time_length: Total time span of observations (days)
    - depth: Transit depth (fractional, e.g., 0.01 for 1%)
    - snr: Desired signal-to-noise ratio for the integrated transit signal
           SNR = total_signal / (sigma * sqrt(n_in_transit))
    - cadence: Sampling cadence in days (default 30 minutes = 1/48 day)
    - seed: Optional random seed for reproducibility

    Assumptions
    - Sun-like star defaults for limb darkening (TESS-like)
    - Single-planet, circular orbit, simple box-like depth controlled via rdr
    - Flux baseline normalized around 1.0
    - SNR computed from integrated signal across in-transit points

    Returns
    - time: Array of observation times (days)
    - flux: Synthetic flux array with noise, baseline ~1
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Build time array
    npts = int(np.floor(time_length / cadence)) + 1
    time = np.arange(npts, dtype=float) * cadence
    
    # Integration time array (assume same as cadence for simplicity)
    itime = np.full_like(time, cadence)

    # Create transit model solution
    sol = transitm.transit_model_class()
    # Limb darkening defaults (TESS-like)
    sol.nl3 = 0.311
    sol.nl4 = 0.270
    sol.rho = kep.rhostar(per, 0.1)  # rough placeholder based on period and dummy duration
    sol.zpt = 0.0
    sol.dil = 0.0
    sol.vof = 0.0

    # Single planet parameters
    sol.npl = 1
    sol.t0  = [t0]
    sol.per = [per]
    sol.bb  = [0.5]
    # Map requested depth to radius ratio rdr: depth ≈ (Rp/R*)^2
    sol.rdr = [float(np.sqrt(abs(depth)))]
    sol.ecw = [0.0]
    sol.esw = [0.0]
    sol.krv = [0.0]
    sol.ted = [0.0]
    sol.ell = [0.0]
    sol.alb = [0.0]

    # Generate noiseless model flux around baseline ~1.0
    # Use transitModel as in plotTransit: tmodel = transitModel(sol, time, itime, nintg, ntt, tobs, omc) - zpt
    nintg = 41  # Number of integration points
    ntt = -1    # No TTVs
    tobs = -1   # No TTV observations
    omc = -1    # No TTV O-C
    
    f_model = transitModel(sol, time, itime, nintg, ntt, tobs, omc) - sol.zpt
    
    # Ensure baseline ~1.0
    baseline = np.median(f_model)
    if baseline <= 0 or not np.isfinite(baseline):
        baseline = 1.0
    f_model = f_model / baseline

    # Calculate SNR based on integrated transit signal
    # Subtract 1 from transit model to get the signal (negative dip)
    signal = 1.0 - f_model
    
    # Integrate the signal across all in-transit points (where signal is significant)
    # Define in-transit as points where signal exceeds 10% of maximum depth
    in_transit = signal > (0.1 * np.max(signal))
    n_in_transit = np.sum(in_transit)
    
    if n_in_transit > 0:
        # Total integrated signal
        total_signal = np.sum(signal[in_transit])
        
        # For SNR = total_signal / (sigma * sqrt(n_in_transit))
        # Solve for sigma: sigma = total_signal / (snr * sqrt(n_in_transit))
        sigma = total_signal / (float(max(snr, 1e-9)) * np.sqrt(n_in_transit))
    else:
        # Fallback if no transit detected
        sigma = float(abs(depth)) / float(max(snr, 1e-9))
    
    noise = rng.normal(0.0, sigma, size=time.shape[0])

    flux = f_model + noise
    return time, flux
