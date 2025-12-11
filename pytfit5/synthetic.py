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
) -> Tuple[np.ndarray, np.ndarray, transitm.transit_model_class]:
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
    - sol: Transit model solution object with injection parameters
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
    # Use solar density (1.4 g/cm^3) as default
    sol.rho = 1.4
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
    return time, flux, sol


def mark_in_transit(time: np.ndarray, t0: float, per: float, duration: float) -> np.ndarray:
    """
    Mark which observations are in transit.
    
    Parameters
    ----------
    time : np.ndarray
        Array of observation times (days)
    t0 : float
        Transit center time (days)
    per : float
        Orbital period (days)
    duration : float
        Transit duration (days)
    
    Returns
    -------
    in_transit : np.ndarray (bool)
        Boolean array marking in-transit observations
    """
    # Calculate phase for each observation
    phase = ((time - t0) % per) / per
    
    # Center phase around transit (phase = 0 at transit center)
    # Handle wrap-around: phases > 0.5 are actually negative phases
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # Convert phase to time from transit center
    dt_from_transit = phase * per
    
    # Mark observations within half-duration of transit center
    in_transit = np.abs(dt_from_transit) <= (duration / 2.0)
    
    return in_transit


def calculate_transit_overlap(
    time: np.ndarray,
    true_t0: float,
    true_per: float,
    true_duration: float,
    recovered_t0: float,
    recovered_per: float,
    recovered_duration: float,
    max_period_factor: int = 5
) -> dict:
    """
    Calculate overlap between true and recovered transit windows.
    
    This function handles period aliases by testing if the recovered period
    is an integer multiple or fraction of the true period.
    
    Parameters
    ----------
    time : np.ndarray
        Array of observation times (days)
    true_t0 : float
        Injected transit center time (days)
    true_per : float
        Injected period (days)
    true_duration : float
        Injected transit duration (days)
    recovered_t0 : float
        BLS detected transit center time (days)
    recovered_per : float
        BLS detected period (days)
    recovered_duration : float
        BLS detected transit duration (days)
    max_period_factor : int, optional
        Maximum integer factor to test for period aliases (default: 5)
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'overlap_fraction': Fraction of true in-transit points recovered (0 to 1)
        - 'precision': Fraction of recovered points that are truly in-transit (0 to 1)
        - 'true_positive': Number of correctly identified in-transit points
        - 'false_positive': Number of incorrectly identified in-transit points
        - 'false_negative': Number of missed in-transit points
        - 'period_factor': Best matching period factor (1 = exact match, 2 = 2x alias, etc.)
        - 'period_factor_type': 'exact', 'multiple', or 'fraction'
        - 'is_recovered': Boolean indicating if transit was successfully recovered
    """
    # Mark true in-transit observations
    true_in_transit = mark_in_transit(time, true_t0, true_per, true_duration)
    n_true = np.sum(true_in_transit)
    
    if n_true == 0:
        # No true transits in dataset
        return {
            'overlap_fraction': 0.0,
            'precision': 0.0,
            'true_positive': 0,
            'false_positive': 0,
            'false_negative': 0,
            'period_factor': 0,
            'period_factor_type': 'none',
            'is_recovered': False
        }
    
    # Test different period factors to find best match
    best_overlap = 0.0
    best_factor = 1
    best_factor_type = 'exact'
    best_recovered_mask = None
    
    # Test if recovered period is a multiple of true period
    for factor in range(1, max_period_factor + 1):
        test_per = true_per * factor
        if np.abs(recovered_per - test_per) / test_per < 0.05:  # Within 5%
            recovered_in_transit = mark_in_transit(time, recovered_t0, recovered_per, recovered_duration)
            overlap = np.sum(true_in_transit & recovered_in_transit) / n_true
            if overlap > best_overlap:
                best_overlap = overlap
                best_factor = factor
                best_factor_type = 'multiple' if factor > 1 else 'exact'
                best_recovered_mask = recovered_in_transit
    
    # Test if recovered period is a fraction of true period
    for factor in range(2, max_period_factor + 1):
        test_per = true_per / factor
        if np.abs(recovered_per - test_per) / test_per < 0.05:  # Within 5%
            recovered_in_transit = mark_in_transit(time, recovered_t0, recovered_per, recovered_duration)
            overlap = np.sum(true_in_transit & recovered_in_transit) / n_true
            if overlap > best_overlap:
                best_overlap = overlap
                best_factor = factor
                best_factor_type = 'fraction'
                best_recovered_mask = recovered_in_transit
    
    # If no good match found, still calculate overlap using recovered parameters
    # This handles cases where period is very close but not exactly matching any factor
    if best_recovered_mask is None:
        # For close period matches (within 10%), use the recovered parameters
        period_ratio = recovered_per / true_per
        if 0.9 < period_ratio < 1.1:
            # Periods are very close, calculate overlap directly
            recovered_in_transit = mark_in_transit(time, recovered_t0, recovered_per, recovered_duration)
            best_overlap = np.sum(true_in_transit & recovered_in_transit) / n_true
            best_recovered_mask = recovered_in_transit
            best_factor = 1
            best_factor_type = 'exact' if 0.99 < period_ratio < 1.01 else 'close'
        else:
            # Period significantly different, still calculate but mark as mismatch
            recovered_in_transit = mark_in_transit(time, recovered_t0, recovered_per, recovered_duration)
            best_overlap = np.sum(true_in_transit & recovered_in_transit) / n_true
            best_recovered_mask = recovered_in_transit
            best_factor_type = 'mismatch'
    
    # Calculate statistics
    true_positive = np.sum(true_in_transit & best_recovered_mask)
    false_positive = np.sum(~true_in_transit & best_recovered_mask)
    false_negative = np.sum(true_in_transit & ~best_recovered_mask)
    n_recovered = np.sum(best_recovered_mask)
    
    overlap_fraction = true_positive / n_true if n_true > 0 else 0.0
    precision = true_positive / n_recovered if n_recovered > 0 else 0.0
    
    # Consider transit recovered if overlap > 50% and precision > 50%
    is_recovered = (overlap_fraction > 0.5) and (precision > 0.5)
    
    return {
        'overlap_fraction': overlap_fraction,
        'precision': precision,
        'true_positive': int(true_positive),
        'false_positive': int(false_positive),
        'false_negative': int(false_negative),
        'period_factor': best_factor,
        'period_factor_type': best_factor_type,
        'is_recovered': is_recovered
    }


def compare_bls_injection(
    time: np.ndarray,
    sol_injected: transitm.transit_model_class,
    sol_bls: transitm.transit_model_class,
    verbose: bool = True
) -> dict:
    """
    Compare BLS detection results with injection parameters.
    
    Convenience wrapper around calculate_transit_overlap with nice output formatting.
    
    Parameters
    ----------
    time : np.ndarray
        Array of observation times (days)
    sol_injected : transit_model_class
        Solution object with injected transit parameters
    sol_bls : transit_model_class
        Solution object with BLS recovered parameters
    verbose : bool, optional
        Print comparison results (default: True)
    
    Returns
    -------
    result : dict
        Recovery statistics (see calculate_transit_overlap)
    """
    # Extract parameters from solution objects
    injected_t0 = sol_injected.t0[0]
    injected_per = sol_injected.per[0]
    injected_duration = kep.transitDuration(sol_injected, i_planet=0)
    
    bls_t0 = sol_bls.t0[0]
    bls_per = sol_bls.per[0]
    bls_duration = kep.transitDuration(sol_bls, i_planet=0)
    
    result = calculate_transit_overlap(
        time, injected_t0, injected_per, injected_duration,
        bls_t0, bls_per, bls_duration
    )
    
    if verbose:
        print("=" * 60)
        print("BLS Recovery Analysis")
        print("=" * 60)
        print(f"Injected Period:    {injected_per:.6f} days")
        print(f"BLS Period:         {bls_per:.6f} days")
        print(f"Period Ratio:       {bls_per/injected_per:.4f}")
        print(f"Period Factor Type: {result['period_factor_type']}")
        if result['period_factor'] > 1:
            print(f"Period Factor:      {result['period_factor']}")
        print()
        print(f"Injected T0:        {injected_t0:.6f} days")
        print(f"BLS T0:             {bls_t0:.6f} days")
        print(f"T0 Difference:      {abs(bls_t0 - injected_t0):.6f} days")
        print()
        print(f"Overlap Fraction:   {result['overlap_fraction']:.2%}")
        print(f"Precision:          {result['precision']:.2%}")
        print(f"True Positives:     {result['true_positive']}")
        print(f"False Positives:    {result['false_positive']}")
        print(f"False Negatives:    {result['false_negative']}")
        print()
        print(f"Recovery Status:    {'✓ RECOVERED' if result['is_recovered'] else '✗ NOT RECOVERED'}")
        print("=" * 60)
    
    return result
