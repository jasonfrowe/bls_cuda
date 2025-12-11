# pytfit5 API Reference

**Version 1.0.0**

This document provides detailed API documentation for the `pytfit5` package, covering Box-Least-Squares (BLS) transit search, transit modeling, and synthetic lightcurve generation.

---

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [BLS Module (`pytfit5.gbls`)](#bls-module-pytfit5gbls)
4. [Transit Model Module (`pytfit5.transitmodel`)](#transit-model-module-pytfit5transitmodel)
5. [Synthetic Lightcurves (`pytfit5.synthetic`)](#synthetic-lightcurves-pytfit5synthetic)
6. [Keplerian Utilities (`pytfit5.kep`)](#keplerian-utilities-pytfit5kep)
7. [MCMC Module (`pytfit5.tmcmc`)](#mcmc-module-pytfit5tmcmc)
8. [Examples](#examples)

---

## Installation

```bash
git clone https://github.com/jasonfrowe/bls_cuda.git
cd bls_cuda
pip install -e .
```

### Dependencies

- `numpy` - Array operations
- `scipy` - Signal processing and optimization
- `matplotlib` - Plotting
- `numba` - JIT compilation for performance
- `tqdm` - Progress bars
- `pandas` - Data handling
- `astroquery` - Astronomical data access

---

## Package Structure

```python
import pytfit5

# Main submodules (accessible via aliases):
pytfit5.gbls   # BLS transit search (bls_cpu module)
pytfit5.tpy5   # Transit fitting (transitPy5 module)
pytfit5.kep    # Keplerian orbit utilities
pytfit5.tmcmc  # MCMC transit fitting
pytfit5.transitmodel  # Transit light curve modeling
pytfit5.occult        # Occultation modeling
pytfit5.effects       # Astrophysical effects (beaming, ellipsoidal, etc.)
pytfit5.synthetic     # Synthetic lightcurve generation
```

---

## BLS Module (`pytfit5.gbls`)

The Box-Least-Squares (BLS) algorithm is used to detect periodic transit signals in photometric time series data.

### Core Classes

#### `gbls_inputs_class`

Configuration class for BLS search parameters.

**Attributes:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | `"filename.txt"` | Input light curve file path |
| `lcdir` | str | `""` | Light curve directory |
| `zerotime` | float | `0.0` | Reference time offset (days) |
| `freq1` | float | `-1` | Minimum search frequency (c/d) |
| `freq2` | float | `-1` | Maximum search frequency (c/d) |
| `ofac` | float | `8.0` | Oversampling factor for frequency grid |
| `Mstar` | float | `1.0` | Stellar mass (solar masses) |
| `Rstar` | float | `1.0` | Stellar radius (solar radii) |
| `nper` | int | `50000` | Maximum number of periods to search |
| `minbin` | int | `5` | Minimum bins in transit |
| `plots` | int | `1` | Plot mode: 0=none, 1=X11, 2=PNG+X11, 3=PNG |
| `multipro` | int | `1` | Enable multiprocessing (0=off, 1=on) |
| `normalize` | str | `"coverage_mad"` | Normalization mode (see below) |

**Normalization Modes:**

- `"none"` - No normalization, raw BLS power
- `"mad"` - Median Absolute Deviation normalization
- `"percentile_mad"` - 75th percentile baseline with MAD noise
- `"coverage_mad"` - MAD weighted by frequency bin coverage
- `"iterative_baseline"` - **Recommended for high-SNR signals** - Uses sigma-clipping to robustly identify continuum, preventing strong peaks from biasing normalization

**Example:**

```python
import pytfit5.gbls as gbls

# Configure BLS search
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 1.0 / 60.0  # Search periods from 1 to 60 days
inputs.freq2 = 2.0
inputs.ofac = 10.0
inputs.Mstar = 1.0
inputs.Rstar = 1.0
inputs.normalize = "iterative_baseline"
```

---

#### `gbls_ans_class`

Results class storing BLS detection information.

**Attributes:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `epo` | float | Best-fit epoch (days) |
| `bper` | float | Best-fit period (days) |
| `bpower` | float | Maximum BLS power |
| `snr` | float | Signal-to-noise ratio |
| `tdur` | float | Transit duration (hours) |
| `depth` | float | Transit depth (fractional) |

---

### Core Functions

#### `bls(time, flux, inputs)`

Main BLS search function.

**Parameters:**

- `time` (np.ndarray): Time array (days)
- `flux` (np.ndarray): Flux array (normalized)
- `inputs` (gbls_inputs_class): BLS configuration

**Returns:**

- `ans` (gbls_ans_class): Detection results
- `freqs` (np.ndarray): Frequency grid (c/d)
- `power` (np.ndarray): BLS power spectrum

**Example:**

```python
import pytfit5.gbls as gbls
import numpy as np

# Load or generate data
time, flux = gbls.readfile("lightcurve.txt")

# Configure search
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 0.05
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"

# Run BLS
ans, freqs, power = gbls.bls(time, flux, inputs)

print(f"Period: {ans.bper:.6f} days")
print(f"Epoch: {ans.epo:.6f} days")
print(f"SNR: {ans.snr:.2f}")
```

---

#### `calc_eph(p, freqs, inputs, time)`

Calculate normalized BLS periodogram with advanced baseline and noise modeling.

**Parameters:**

- `p` (np.ndarray): Raw BLS power array
- `freqs` (np.ndarray): Frequency grid (c/d)
- `inputs` (gbls_inputs_class): Configuration
- `time` (np.ndarray): Time array for extrapolation

**Returns:**

- `power` (np.ndarray): Normalized BLS power spectrum

**Features:**

- **Duty-cycle informed filtering**: Adapts kernel width based on expected transit duration
- **Baseline extrapolation**: For periods > 0.3×T_baseline, extrapolates rolling median baseline from well-sampled regions to prevent long-period suppression
- **Noise extrapolation**: Maintains proper noise floor at low frequencies
- **Multiple normalization modes**: Robust to high-SNR signals and varying coverage

---

#### `readfile(filename)`

Read two-column light curve files (time, flux).

**Parameters:**

- `filename` (str): Path to ASCII file

**Returns:**

- `time` (np.ndarray): Time array
- `flux` (np.ndarray): Flux array

**File Format:**

```
# Header line (ignored)
2459000.1234  0.9998
2459000.1445  1.0002
2459000.1656  0.9995
...
```

---

### Advanced Features

#### Baseline and Noise Extrapolation

For long-period signals (approaching dataset length), the BLS normalization uses extrapolation to prevent suppression:

1. **Threshold Detection**: Identifies low-frequency bins where periods > 0.5 × T_baseline (need at least 2 full periods)
2. **Linear Fit**: Fits baseline and noise vs. log(frequency) in well-sampled region (periods < 0.25 × T_baseline)
3. **Extrapolation**: Applies fitted trend to low-frequency region

This ensures long-period transits maintain proper SNR in the normalized periodogram.

---

## Transit Model Module (`pytfit5.transitmodel`)

### Core Classes

#### `transit_model_class`

Complete transit model solution container.

**Key Attributes:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `npl` | int | Number of planets |
| `t0` | list[float] | Transit center times (days) |
| `per` | list[float] | Orbital periods (days) |
| `bb` | list[float] | Impact parameters |
| `rdr` | list[float] | Radius ratios (Rp/R*) |
| `ecw` | list[float] | e·cos(ω) eccentricity components |
| `esw` | list[float] | e·sin(ω) eccentricity components |
| `nl3` | float | Quadratic limb darkening coefficient u1 |
| `nl4` | float | Quadratic limb darkening coefficient u2 |
| `rho` | float | Stellar density (g/cm³) |
| `zpt` | float | Flux zero point |
| `dil` | float | Dilution factor |

---

### Core Functions

#### `transitModel(sol, time, itime, nintg=41, ntt=-1, tobs=-1, omc=-1)`

Generate transit light curve from model parameters.

**Parameters:**

- `sol` (transit_model_class): Model parameters
- `time` (np.ndarray): Time array (days)
- `itime` (np.ndarray): Integration time per point (days)
- `nintg` (int): Number of integration steps (default: 41)
- `ntt` (int): Number of time points (default: -1, auto-detect)
- `tobs` (float): Observation time (default: -1, auto-detect)
- `omc` (float): O-C correction (default: -1, unused)

**Returns:**

- `model` (np.ndarray): Model flux array

**Example:**

```python
import pytfit5.transitmodel as tm
import numpy as np

# Create model
sol = tm.transit_model_class()
sol.npl = 1
sol.t0 = [2459000.5]
sol.per = [3.5]
sol.bb = [0.3]
sol.rdr = [0.1]  # 10% radius ratio
sol.ecw = [0.0]
sol.esw = [0.0]
sol.nl3 = 0.311  # TESS-like limb darkening
sol.nl4 = 0.270
sol.rho = 1.4  # Solar-like density

# Generate time array
time = np.linspace(2459000.0, 2459010.0, 1000)
itime = np.full_like(time, 30.0/1440.0)  # 30-minute cadence

# Compute model
model = tm.transitModel(sol, time, itime, nintg=41)
```

---

## Synthetic Lightcurves (`pytfit5.synthetic`)

Utilities for generating synthetic transit lightcurves for BLS testing and validation.

### Functions

#### `generate_synthetic_lightcurve(t0, per, time_length, depth, snr, cadence=1/48, seed=None)`

Generate realistic synthetic transit lightcurve with controlled SNR.

**Parameters:**

- `t0` (float): Transit center time (days)
- `per` (float): Orbital period (days)
- `time_length` (float): Total observation duration (days)
- `depth` (float): Transit depth (fractional, e.g., 0.01 = 1%)
- `snr` (float): Desired signal-to-noise ratio (integrated over transit)
- `cadence` (float): Sampling interval (days, default: 1/48 = 30 min)
- `seed` (int, optional): Random seed for reproducibility

**Returns:**

- `time` (np.ndarray): Time array (days)
- `flux` (np.ndarray): Flux array with noise, baseline ~1.0

**SNR Definition:**

$$\text{SNR} = \frac{\sum_{i \in \text{transit}} |1 - f_i|}{\sigma \sqrt{N_{\text{transit}}}}$$

where $\sigma$ is the noise level, computed to achieve the requested SNR.

**Example:**

```python
from pytfit5.synthetic import generate_synthetic_lightcurve

# Generate 60-day lightcurve with 20-day period, SNR=10
# Generate synthetic data
time, flux, sol_injected = generate_synthetic_lightcurve(
    t0=5.0,
    per=20.05,
    time_length=60.0,
    depth=0.01,  # 1% transit
    snr=10.0,
    cadence=1/48,
    seed=42
)

# Run BLS on synthetic data
import pytfit5.gbls as gbls

inputs = gbls.gbls_inputs_class()
inputs.freq1 = 1.0/60.0
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"

ans, freqs, power = gbls.bls(time, flux, inputs)
```

---

#### `compare_bls_injection(time, sol_injected, sol_bls, verbose=True)`

Compare BLS detection results with injection parameters and calculate recovery statistics.

**Parameters:**

- `time` (np.ndarray): Array of observation times (days)
- `sol_injected` (transit_model_class): Solution object with injected transit parameters
- `sol_bls` (transit_model_class): Solution object with BLS recovered parameters
- `verbose` (bool): Print comparison results (default: True)

**Returns:**

- `result` (dict): Recovery statistics containing:
  - `overlap_fraction`: Fraction of true in-transit points recovered (0-1)
  - `precision`: Fraction of recovered points that are truly in-transit (0-1)
  - `true_positive`: Number of correctly identified in-transit points
  - `false_positive`: Number of incorrectly identified in-transit points
  - `false_negative`: Number of missed in-transit points
  - `period_factor`: Best matching period factor (1 = exact, 2 = 2× alias, etc.)
  - `period_factor_type`: 'exact', 'multiple', 'fraction', or 'mismatch'
  - `is_recovered`: Boolean indicating successful recovery (overlap > 50% and precision > 50%)

**Features:**

- Automatically tests for period aliases (integer multiples and fractions up to 5×)
- Handles phase-shifted detections
- Calculates overlap between true and recovered transit windows
- Provides detailed recovery statistics

**Example:**

```python
from pytfit5.synthetic import generate_synthetic_lightcurve, compare_bls_injection
import pytfit5.gbls as gbls
import pytfit5.transitmodel as transitm

# Generate synthetic data (returns solution object)
time, flux, sol_injected = generate_synthetic_lightcurve(
    t0=5.0, per=10.5, time_length=60.0, depth=0.01, snr=15.0, seed=42
)

# Run BLS
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 0.05
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"
ans = gbls.bls(inputs, time, flux)

# Create solution object from BLS results
sol_bls = transitm.transit_model_class()
sol_bls.npl = 1
sol_bls.t0 = [ans.epo]
sol_bls.per = [ans.bper]
sol_bls.bb = [0.5]  # Assumed impact parameter
sol_bls.rdr = [np.sqrt(ans.depth)]
sol_bls.rho = 1.4  # Assumed stellar density (g/cm³)

# Compare results using solution objects
recovery = compare_bls_injection(
    time=time,
    sol_injected=sol_injected,
    sol_bls=sol_bls,
    verbose=True
)

print(f"Recovery success: {recovery['is_recovered']}")
print(f"Overlap: {recovery['overlap_fraction']:.1%}")
```

---

#### `calculate_transit_overlap(time, true_t0, true_per, true_duration, recovered_t0, recovered_per, recovered_duration, max_period_factor=5)`

Lower-level function to calculate overlap between true and recovered transit windows.

**Parameters:**

- `time` (np.ndarray): Observation times
- `true_t0`, `true_per`, `true_duration` (float): Injected transit parameters
- `recovered_t0`, `recovered_per`, `recovered_duration` (float): BLS results
- `max_period_factor` (int): Maximum integer factor for period aliases (default: 5)

**Returns:**

- `result` (dict): Same statistics as `compare_bls_injection` but without verbose output

---

#### `mark_in_transit(time, t0, per, duration)`

Mark which observations fall within transit windows.

**Parameters:**

- `time` (np.ndarray): Observation times (days)
- `t0` (float): Transit center time (days)
- `per` (float): Orbital period (days)
- `duration` (float): Transit duration (days)

**Returns:**

- `in_transit` (np.ndarray): Boolean array marking in-transit observations

**Example:**

```python
from pytfit5.synthetic import mark_in_transit

in_transit = mark_in_transit(time, t0=5.0, per=10.5, duration=0.1)
print(f"In-transit points: {np.sum(in_transit)}")
```

---

## Keplerian Utilities (`pytfit5.kep`)

Orbital mechanics and stellar parameter utilities.

### Functions

#### `rhostar(per, duration)`

Calculate stellar density from period and transit duration.

**Parameters:**

- `per` (float): Orbital period (days)
- `duration` (float): Transit duration (days)

**Returns:**

- `rho` (float): Stellar density (g/cm³)

**Formula:**

$$\rho_\star = \frac{3\pi}{G P^2} \left(\frac{\tau}{P}\right)^{3}$$

---

## MCMC Module (`pytfit5.tmcmc`)

Markov Chain Monte Carlo transit fitting for parameter estimation and uncertainty quantification.

### Functions

#### `run_mcmc(time, flux, ferr, initial_params, ...)`

Run MCMC analysis on transit lightcurve.

**Parameters:**

- `time` (np.ndarray): Time array
- `flux` (np.ndarray): Flux array
- `ferr` (np.ndarray): Flux uncertainties
- `initial_params` (dict): Starting parameter values

**Returns:**

- `samples` (np.ndarray): MCMC chain samples
- `best_fit` (dict): Maximum likelihood parameters

---

## Examples

### Example 1: Basic BLS Search

```python
import pytfit5.gbls as gbls
import numpy as np

# Load data
time, flux = gbls.readfile("star_lightcurve.txt")

# Configure BLS
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 0.1  # 10-day maximum period
inputs.freq2 = 4.0  # 0.25-day minimum period
inputs.ofac = 10.0
inputs.Mstar = 0.8
inputs.Rstar = 0.75
inputs.normalize = "iterative_baseline"

# Search for transits
ans, freqs, power = gbls.bls(time, flux, inputs)

print(f"Detected Period: {ans.bper:.4f} days")
print(f"Transit Epoch: {ans.epo:.4f}")
print(f"SNR: {ans.snr:.1f}")
print(f"Depth: {ans.depth*1e6:.1f} ppm")
```

### Example 2: High-SNR Synthetic Test

```python
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.gbls as gbls
import matplotlib.pyplot as plt

# Generate high-SNR synthetic data
time, flux = generate_synthetic_lightcurve(
    t0=2.5,
    per=8.3,
    time_length=100.0,
    depth=0.02,  # 2% deep transit
    snr=200.0,   # Very high SNR
    seed=123
)

# BLS with iterative baseline (robust to high SNR)
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 0.01
inputs.freq2 = 2.0
inputs.normalize = "iterative_baseline"

ans, freqs, power = gbls.bls(time, flux, inputs)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(time, flux, 'k.', ms=2)
plt.xlabel('Time (days)')
plt.ylabel('Flux')

plt.subplot(122)
plt.plot(1.0/freqs, power, 'b-', lw=0.5)
plt.axvline(ans.bper, color='r', ls='--', label=f'Detected: {ans.bper:.2f} d')
plt.axvline(8.3, color='g', ls='--', label='True: 8.30 d')
plt.xlabel('Period (days)')
plt.ylabel('BLS Power')
plt.legend()
plt.tight_layout()
plt.show()
```

### Example 3: Long-Period Search with Extrapolation

```python
from pytfit5.synthetic import generate_synthetic_lightcurve
import pytfit5.gbls as gbls

# Generate long-period transit in short baseline
time, flux = generate_synthetic_lightcurve(
    t0=5.0,
    per=45.0,  # Period close to baseline length
    time_length=60.0,
    depth=0.005,
    snr=15.0,
    seed=99
)

# BLS will extrapolate baseline for periods > 0.5*60 = 30 days
inputs = gbls.gbls_inputs_class()
inputs.freq1 = 1.0/60.0
inputs.freq2 = 1.0
inputs.normalize = "iterative_baseline"

ans, freqs, power = gbls.bls(time, flux, inputs)

print(f"True period: 45.0 days")
print(f"Recovered period: {ans.bper:.2f} days")
print(f"SNR: {ans.snr:.1f}")
```

---

## Physical Constants

The following constants are defined in `pytfit5.gbls`:

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| `G` | 6.674×10⁻¹¹ | N·m²/kg² | Gravitational constant |
| `Rsun` | 6.96265×10⁸ | m | Solar radius |
| `Msun` | 1.9891×10³⁰ | kg | Solar mass |
| `pifac` | 1.083852 | - | (2π)^(2/3)/π |
| `day2sec` | 86400 | s/day | Seconds per day |
| `onehour` | 1/24 | day/hr | Days per hour |

---

## Performance Notes

- **Numba JIT**: Core BLS kernel functions are JIT-compiled for ~100× speedup
- **Multiprocessing**: Set `inputs.multipro = 1` for parallel processing
- **Frequency Grid**: Higher `ofac` values improve frequency resolution but increase computation time
- **Normalization**: `"iterative_baseline"` mode adds ~10-20% overhead but significantly improves robustness

---

## License

GNU General Public License v3.0 (GPL-3.0)

---

## Citation

If you use this software in your research, please cite:

```
Rowe, J. F. et al. (2025). pytfit5: Python Implementation of TransitFit5 and BLS.
https://github.com/jasonfrowe/bls_cuda
```

---

## Support

- **Issues**: https://github.com/jasonfrowe/bls_cuda/issues
- **Documentation**: See `pytfit5_example.ipynb` for interactive examples
- **Contact**: jason@jasonrowe.org
