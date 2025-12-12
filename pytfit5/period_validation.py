"""
Period validation tools for BLS results.

This module provides functions to validate BLS-reported periods by:
1. Testing for period aliases (factors of 2, 3, 4, 5, etc.)
2. Checking for single-transit events
3. Computing SNR for different period hypotheses
4. Optimizing epoch (t0) for each period hypothesis
"""

import numpy as np
from numba import jit
import math


def get_stellar_density(mass_solar, radius_solar):
    """
    Calculates mean stellar density given mass and radius in solar units.
    
    Args:
        mass_solar (float): Mass of the star in Solar Masses (M_sun).
        radius_solar (float): Radius of the star in Solar Radii (R_sun).
        
    Returns:
        float: Mean density in g/cm^3.
    """
    # Constants
    M_SUN_KG = 1.98847e30  # Mass of Sun in kg
    R_SUN_M = 6.957e8      # Radius of Sun in meters
    
    # Convert inputs to SI units
    mass_kg = mass_solar * M_SUN_KG
    radius_m = radius_solar * R_SUN_M
    
    # Calculate Volume of a sphere (4/3 * pi * r^3)
    volume = (4/3) * math.pi * (radius_m ** 3)
    
    # Calculate Density (Mass / Volume)
    density = mass_kg / volume
    
    return density / 1000.0

@jit(nopython=True)
def compute_transit_snr(time, flux, t0, period, tdur, n_dur=1.5):
    """
    Compute SNR for a given period and epoch by identifying in-transit points.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    flux : np.ndarray
        Flux values (normalized around 0)
    t0 : float
        Transit epoch (days)
    period : float
        Orbital period (days)
    tdur : float
        Transit duration (days)
    n_dur : float
        Number of transit durations to include (default 1.5 for safety)
        
    Returns
    -------
    snr : float
        Signal-to-noise ratio of the transit
    depth : float
        Transit depth (out-of-transit median - in-transit median)
    n_transits : int
        Number of transits detected
    in_transit_mask : np.ndarray
        Boolean mask of in-transit points
    """
    npt = len(time)
    half_dur = n_dur * tdur / 2.0
    
    # Phase fold the data
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # Center on 0
    
    # Identify in-transit points
    in_transit_mask = np.abs(phase * period) < half_dur
    out_transit_mask = ~in_transit_mask
    
    n_in = np.sum(in_transit_mask)
    n_out = np.sum(out_transit_mask)
    
    if n_in < 3 or n_out < 3:
        return 0.0, 0.0, 0, in_transit_mask
    
    # Compute statistics
    flux_in = flux[in_transit_mask]
    flux_out = flux[out_transit_mask]
    
    depth = np.median(flux_out) - np.median(flux_in)
    noise = np.std(flux_out)
    
    if noise <= 0:
        return 0.0, depth, 0, in_transit_mask
    
    snr = depth / noise * np.sqrt(n_in)
    
    # Count number of transits (unique transit events)
    transit_numbers = np.floor((time[in_transit_mask] - t0) / period + 0.5)
    n_transits = len(np.unique(transit_numbers))
    
    return snr, depth, n_transits, in_transit_mask


def analyze_individual_transits(time, flux, t0, period, tdur, n_dur=1.5):
    """
    Analyze SNR of each individual transit to detect single-transit events.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    flux : np.ndarray
        Flux values (normalized around 0)
    t0 : float
        Transit epoch (days)
    period : float
        Orbital period (days)
    tdur : float
        Transit duration (days)
    n_dur : float
        Number of transit durations to include
        
    Returns
    -------
    transit_info : dict
        Dictionary containing:
        - transit_numbers: Array of transit indices
        - transit_times: Array of transit times
        - transit_snrs: Array of individual transit SNRs
        - transit_depths: Array of individual transit depths
        - strongest_snr: SNR of strongest transit
        - mean_snr: Mean SNR of all transits
        - median_snr: Median SNR of all transits
        - std_snr: Standard deviation of SNRs
        - is_single_dominant: Boolean indicating if one transit dominates
    """
    # Get overall in-transit mask
    _, _, _, in_transit_mask = compute_transit_snr(time, flux, t0, period, tdur, n_dur)
    
    # Phase fold and identify transit numbers
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    transit_number = np.round((time - t0) / period).astype(int)
    
    # Get unique transits
    unique_transits = np.unique(transit_number)
    
    transit_numbers = []
    transit_times = []
    transit_snrs = []
    transit_depths = []
    
    # Calculate noise from out-of-transit data
    out_transit_flux = flux[~in_transit_mask]
    if len(out_transit_flux) > 0:
        noise = np.std(out_transit_flux)
    else:
        noise = np.std(flux)
    
    # Analyze each transit individually
    for tr_num in unique_transits:
        # Mask for this specific transit
        mask = (transit_number == tr_num) & (np.abs(phase) < tdur/(2*period))
        
        if np.sum(mask) > 0:
            transit_time = t0 + tr_num * period
            in_transit_flux = flux[mask]
            
            if len(in_transit_flux) > 0:
                depth = 1.0 - np.median(in_transit_flux)
                snr_single = depth * np.sqrt(len(in_transit_flux)) / noise if noise > 0 else 0
                
                transit_numbers.append(tr_num)
                transit_times.append(transit_time)
                transit_snrs.append(snr_single)
                transit_depths.append(depth)
    
    # Convert to arrays
    transit_numbers = np.array(transit_numbers)
    transit_times = np.array(transit_times)
    transit_snrs = np.array(transit_snrs)
    transit_depths = np.array(transit_depths)
    
    # Calculate statistics
    if len(transit_snrs) > 0:
        strongest_snr = np.max(transit_snrs)
        mean_snr = np.mean(transit_snrs)
        median_snr = np.median(transit_snrs)
        std_snr = np.std(transit_snrs)
        
        # Check if one transit dominates (strongest is >2.5× the median of others)
        other_snrs = transit_snrs[transit_snrs != strongest_snr]
        if len(other_snrs) > 0:
            is_single_dominant = strongest_snr > 2.5 * np.median(other_snrs)
        else:
            is_single_dominant = True  # Only one transit
    else:
        strongest_snr = 0.0
        mean_snr = 0.0
        median_snr = 0.0
        std_snr = 0.0
        is_single_dominant = False
    
    return {
        'transit_numbers': transit_numbers,
        'transit_times': transit_times,
        'transit_snrs': transit_snrs,
        'transit_depths': transit_depths,
        'strongest_snr': strongest_snr,
        'mean_snr': mean_snr,
        'median_snr': median_snr,
        'std_snr': std_snr,
        'is_single_dominant': is_single_dominant,
        'n_transits': len(transit_snrs)
    }


def optimize_epoch(time, flux, period, tdur, t0_initial, search_window=None):
    """
    Optimize the epoch (t0) for a given period by maximizing SNR.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    flux : np.ndarray
        Flux values (normalized around 0)
    period : float
        Orbital period (days)
    tdur : float
        Transit duration (days)
    t0_initial : float
        Initial guess for epoch (days)
    search_window : float, optional
        Window around t0_initial to search (days). Default is period/2
        
    Returns
    -------
    t0_best : float
        Best-fit epoch
    snr_best : float
        SNR at best epoch
    depth_best : float
        Depth at best epoch
    n_transits : int
        Number of transits at best epoch
    """
    if search_window is None:
        search_window = period / 2.0
    
    # Search over a grid of epochs
    n_steps = max(20, int(search_window / tdur))
    t0_grid = np.linspace(t0_initial - search_window/2, 
                          t0_initial + search_window/2, 
                          n_steps)
    
    snr_best = -np.inf
    t0_best = t0_initial
    depth_best = 0.0
    n_transits_best = 0
    
    for t0_test in t0_grid:
        snr, depth, n_transits, _ = compute_transit_snr(time, flux, t0_test, period, tdur)
        if snr > snr_best:
            snr_best = snr
            t0_best = t0_test
            depth_best = depth
            n_transits_best = n_transits
    
    return t0_best, snr_best, depth_best, n_transits_best


def estimate_single_transit_period(time, tdur, rho_star=1.41, margin_factor=1.5):
    """
    Estimate the orbital period for a single-transit event.
    
    Uses stellar density and transit duration to estimate period assuming
    circular orbit (e=0) and central transit (b=0). If the estimated period
    is short enough that a second transit would be visible in the dataset,
    returns the minimum period that ensures only one transit is visible.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    tdur : float
        Transit duration from BLS (days)
    rho_star : float, optional
        Mean stellar density (g/cm³). Default is 1.41 (solar value)
    margin_factor : float, optional
        Safety margin factor for minimum period calculation (default 1.5)
        Ensures the second transit would occur margin_factor × tdur after
        the end of observations
        
    Returns
    -------
    period_estimate : float
        Estimated orbital period (days)
    period_min : float
        Minimum period consistent with single transit (days)
    is_constrained : bool
        True if period is constrained by dataset length,
        False if using stellar density estimate
    n_transits_expected : float
        Expected number of transits at estimated period
        
    Notes
    -----
    The stellar density-based estimate uses the approximation:
        rho_star ≈ 3*P / (π² * G * tdur³)
    
    Solving for P (assuming b=0, e=0):
        P = π² * G * rho_star * tdur³ / 3
        
    where G = 6.674e-11 m³/(kg·s²)
    
    Examples
    --------
    >>> time = np.linspace(0, 27, 1000)  # 27-day baseline
    >>> tdur = 0.1  # 2.4 hour transit
    >>> P_est, P_min, constrained, n_exp = estimate_single_transit_period(
    ...     time, tdur, rho_star=1.41)
    >>> print(f"Estimated period: {P_est:.1f} days")
    >>> print(f"Minimum period: {P_min:.1f} days")
    >>> print(f"Constrained by data: {constrained}")
    """
    # Physical constants
    G = 6.674e-11  # m³/(kg·s²)
    
    # Convert to SI units
    tdur_sec = tdur * 86400.0  # days to seconds
    rho_star_SI = rho_star * 1000.0  # g/cm³ to kg/m³
    
    # Estimate period from stellar density (assuming b=0, e=0)
    # P = π² * G * rho_star * tdur³ / 3
    period_from_rho = (np.pi**2 * G * rho_star_SI * tdur_sec**3 / 3.0) / 86400.0  # Convert to days
    
    # Calculate dataset properties
    baseline = np.max(time) - np.min(time)
    t_start = np.min(time)
    t_end = np.max(time)
    
    # Minimum period such that only one transit is visible
    # The second transit should occur at least margin_factor * tdur after the end of observations
    # If first transit is at t0, second transit at t0 + P
    # We need: t0 + P > t_end + margin_factor * tdur
    # The latest possible t0 is t_end (transit at end of dataset)
    # So: P_min = margin_factor * tdur (conservative)
    # But we also need: earliest_t0 + P > t_end + margin_factor * tdur
    # If earliest transit at t_start: P_min = baseline + margin_factor * tdur
    period_min = baseline + margin_factor * tdur
    
    # Check if stellar density estimate would result in multiple transits
    if period_from_rho <= period_min:
        # Stellar density estimate predicts multiple transits
        # Use minimum period that ensures single transit
        period_estimate = period_min
        is_constrained = True
        n_transits_expected = baseline / period_estimate
    else:
        # Stellar density estimate is consistent with single transit
        period_estimate = period_from_rho
        is_constrained = False
        n_transits_expected = baseline / period_estimate
    
    return period_estimate, period_min, is_constrained, n_transits_expected


def test_period_aliases(time, flux, t0_bls, period_bls, tdur_bls, 
                        test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                        optimize_t0=True):
    """
    Test if the BLS period is an alias by checking multiples and fractions.
    
    This function tests whether the reported period is actually a harmonic
    or sub-harmonic of the true period. For each test period, it optimizes
    the epoch and computes the SNR.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    flux : np.ndarray
        Flux values (normalized around 0)
    t0_bls : float
        BLS-reported epoch (days)
    period_bls : float
        BLS-reported period (days)
    tdur_bls : float
        BLS-reported transit duration (days)
    test_factors : list of float
        Period factors to test. Default tests P/2, P, 2P, 3P, 4P, 5P
    optimize_t0 : bool
        Whether to optimize epoch for each period (default True)
        
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'factors': array of period factors tested
        - 'periods': array of test periods
        - 't0s': array of optimized epochs
        - 'snrs': array of SNR values
        - 'depths': array of transit depths
        - 'n_transits': array of number of transits
        - 'best_factor': factor with highest SNR
        - 'best_period': period with highest SNR
        - 'best_t0': epoch with highest SNR
        - 'best_snr': highest SNR found
        - 'is_alias': True if best period is not the BLS period
        - 'is_single_transit': True if only 1 transit found at best period
    """
    results = {
        'factors': np.array(test_factors),
        'periods': np.array([period_bls * f for f in test_factors]),
        't0s': [],
        'snrs': [],
        'depths': [],
        'n_transits': []
    }
    
    for factor in test_factors:
        period_test = period_bls * factor
        
        # Initial epoch guess: adjust for period change
        if factor >= 1.0:
            # For longer periods, keep the same initial transit
            t0_initial = t0_bls
        else:
            # For shorter periods, find the first transit within the BLS epoch window
            t0_initial = t0_bls % period_test
        
        if optimize_t0:
            # Optimize epoch for this period
            t0_opt, snr, depth, n_transits = optimize_epoch(
                time, flux, period_test, tdur_bls, t0_initial
            )
        else:
            # Just evaluate at initial epoch
            snr, depth, n_transits, _ = compute_transit_snr(
                time, flux, t0_initial, period_test, tdur_bls
            )
            t0_opt = t0_initial
        
        results['t0s'].append(t0_opt)
        results['snrs'].append(snr)
        results['depths'].append(depth)
        results['n_transits'].append(n_transits)
    
    # Convert to arrays
    results['t0s'] = np.array(results['t0s'])
    results['snrs'] = np.array(results['snrs'])
    results['depths'] = np.array(results['depths'])
    results['n_transits'] = np.array(results['n_transits'])
    
    # Find best result
    best_idx = np.argmax(results['snrs'])
    results['best_factor'] = results['factors'][best_idx]
    results['best_period'] = results['periods'][best_idx]
    results['best_t0'] = results['t0s'][best_idx]
    results['best_snr'] = results['snrs'][best_idx]
    results['best_depth'] = results['depths'][best_idx]
    results['best_n_transits'] = results['n_transits'][best_idx]
    
    # Check if it's an alias (best period is not the BLS period)
    # Use 1% tolerance for floating point comparison
    results['is_alias'] = not np.isclose(results['best_factor'], 1.0, rtol=0.01)
    
    # Check if it's a single-transit event
    results['is_single_transit'] = (results['best_n_transits'] == 1)
    
    # Calculate SNR improvement over BLS period
    bls_idx = np.argmin(np.abs(results['factors'] - 1.0))
    if results['snrs'][bls_idx] > 0:
        results['snr_improvement'] = results['best_snr'] / results['snrs'][bls_idx]
    else:
        results['snr_improvement'] = np.inf
    
    return results


def print_period_validation_report(results, verbose=True, transit_info=None):
    """
    Print a human-readable report of period validation results.
    
    Parameters
    ----------
    results : dict
        Output from test_period_aliases()
    verbose : bool
        If True, print detailed results for all tested periods
    transit_info : dict, optional
        Output from analyze_individual_transits() for detailed transit analysis
    """
    print("\n" + "="*70)
    print("PERIOD VALIDATION REPORT")
    print("="*70)
    
    if verbose:
        print(f"\nTested {len(results['factors'])} period hypotheses:")
        print(f"{'Factor':>8} {'Period (d)':>12} {'Epoch (d)':>12} {'SNR':>8} {'N_transits':>11} {'Depth':>10}")
        print("-"*70)
        for i in range(len(results['factors'])):
            marker = " *** BEST ***" if i == np.argmax(results['snrs']) else ""
            print(f"{results['factors'][i]:>8.1f} {results['periods'][i]:>12.6f} "
                  f"{results['t0s'][i]:>12.6f} {results['snrs'][i]:>8.2f} "
                  f"{results['n_transits'][i]:>11d} {results['depths'][i]:>10.6f}{marker}")
    
    print("\n" + "-"*70)
    print("BEST PERIOD HYPOTHESIS:")
    print("-"*70)
    print(f"  Period factor: {results['best_factor']:.1f}×")
    print(f"  Period:        {results['best_period']:.6f} days")
    print(f"  Epoch:         {results['best_t0']:.6f} days")
    print(f"  SNR:           {results['best_snr']:.2f}")
    print(f"  Depth:         {results['best_depth']:.6f} ({results['best_depth']*1e6:.1f} ppm)")
    print(f"  N transits:    {results['best_n_transits']}")
    
    if results['snr_improvement'] > 1.0:
        print(f"\n  SNR improvement over BLS period: {results['snr_improvement']:.2f}×")
    
    # Print individual transit analysis if provided
    if transit_info is not None and transit_info['n_transits'] > 1:
        print("\n" + "-"*70)
        print("INDIVIDUAL TRANSIT ANALYSIS:")
        print("-"*70)
        print(f"  Strongest transit SNR: {transit_info['strongest_snr']:.2f}")
        print(f"  Mean transit SNR:      {transit_info['mean_snr']:.2f}")
        print(f"  Median transit SNR:    {transit_info['median_snr']:.2f}")
        print(f"  Std dev of SNRs:       {transit_info['std_snr']:.2f}")
        
        if transit_info['is_single_dominant']:
            ratio = transit_info['strongest_snr'] / transit_info['median_snr']
            print(f"\n  ⚠ WARNING: Single dominant transit detected!")
            print(f"     Strongest transit is {ratio:.1f}× the median SNR")
            print(f"     This may be a single-transit event with incorrect period")
    
    print("\n" + "-"*70)
    print("DIAGNOSIS:")
    print("-"*70)
    
    if results['is_single_transit']:
        print("  ⚠️  SINGLE-TRANSIT EVENT")
        print("      Only one transit detected. Period is not well constrained.")
        print("      Additional observations needed for confirmation.")
    elif results['is_alias']:
        if results['best_factor'] < 1.0:
            print(f"  ⚠️  PERIOD ALIAS DETECTED")
            print(f"      BLS period appears to be {1/results['best_factor']:.0f}× the true period.")
            print(f"      True period is likely {results['best_period']:.6f} days.")
        elif results['best_factor'] > 1.0:
            print(f"  ⚠️  PERIOD ALIAS DETECTED")
            print(f"      BLS period appears to be 1/{results['best_factor']:.0f}× the true period.")
            print(f"      True period is likely {results['best_period']:.6f} days.")
        print(f"      Recommended to use factor={results['best_factor']:.1f}× period.")
    else:
        print("  ✓ BLS period appears correct (no alias detected)")
        print(f"    {results['best_n_transits']} transits detected with SNR={results['best_snr']:.2f}")
    
    print("="*70 + "\n")


def validate_bls_period(time, flux, bls_result, 
                       test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                       optimize_t0=True, verbose=True, analyze_transits=True):
    """
    Convenience function to validate a BLS result.
    
    Parameters
    ----------
    time : np.ndarray
        Observation times (days)
    flux : np.ndarray
        Flux values (normalized around 0)
    bls_result : gbls_ans_class
        BLS result object with epo, bper, tdur attributes
    test_factors : list of float
        Period factors to test
    optimize_t0 : bool
        Whether to optimize epoch for each period
    verbose : bool
        Whether to print detailed report
    analyze_transits : bool
        Whether to analyze individual transit SNRs
        
    Returns
    -------
    results : dict
        Validation results from test_period_aliases()
    transit_info : dict or None
        Individual transit analysis if analyze_transits=True
    """
    results = test_period_aliases(
        time, flux, 
        bls_result.epo, 
        bls_result.bper, 
        bls_result.tdur,
        test_factors=test_factors,
        optimize_t0=optimize_t0
    )
    
    # Analyze individual transits for the best period
    transit_info = None
    if analyze_transits and results['best_n_transits'] > 1:
        transit_info = analyze_individual_transits(
            time, flux,
            results['best_t0'],
            results['best_period'],
            bls_result.tdur
        )
        
        # Update single transit detection based on individual analysis
        if transit_info['is_single_dominant']:
            results['is_single_transit'] = True
    
    if verbose:
        print_period_validation_report(results, verbose=True, transit_info=transit_info)
    
    return results, transit_info
