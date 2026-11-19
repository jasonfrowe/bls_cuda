# pytfit5

**Transit search and fitting in Python: Box Least Squares (BLS) with Numba and a compiled C/OpenMP
kernel, and a Python implementation of TransitFit5.**

The package is installed as `pytfit5` from this repository (`bls_cuda`). It provides:

- a periodic BLS search, with a compiled C/OpenMP kernel (AVX-512/AVX2 chosen at run time) and a Numba fallback;
- a single-event ("pulse") search for one-off transits;
- detrending and outlier rejection for long light curves;
- a transit model (Mandel & Agol with Kipping limb-darkening parametrisation, eccentric orbits, TTVs), least-squares fitting and DE-MCMC;
- synthetic light curves and injection/recovery tools.

The original CUDA (GPU) BLS scripts are kept in `cudaFiles/` for reference; they are not part of the
installed package.

## Installation

```bash
git clone https://github.com/jasonfrowe/bls_cuda.git
cd bls_cuda
pip install .
```

Requirements: Python 3.8+, and `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`, `pandas`, `astroquery`
(installed automatically). Optional: `h5py` (saving/loading MCMC results), `exotic_ld` (limb-darkening
coefficients in the example notebook), `pytest` (tests).

### The compiled BLS kernel

`pip install` also tries to build the C/OpenMP BLS kernel (`pytfit5._cbls`). This needs a C compiler with
OpenMP (gcc or clang with libomp on Linux; MSVC on Windows). The build is optional: if it fails, the
package installs anyway and BLS uses the Numba kernel. Apple clang has no OpenMP, so on macOS the kernel
builds single-threaded and `bls_engine="auto"` keeps the (parallel) Numba kernel.

Check what you got:

```python
from pytfit5 import bls_c
print(bls_c.available(), bls_c.has_openmp(), bls_c.supported_variants(), bls_c.resolve_variant())
# True True ['scalar', 'avx2_div', 'avx2_nodiv', 'avx512_div', 'avx512_nodiv'] avx512_nodiv
```

Both kernels give bit-identical periodograms. The C kernel searched a *Kepler* light curve 1.9 times
faster than Numba on an Intel Skylake-SP node (48 threads) and 4.0 times faster on an AMD EPYC 9654
(Zen 4) node (192 threads). The fastest SIMD variant depends on the CPU, so `bls_isa="auto"` picks
`avx512_div` on AMD and `avx512_nodiv` on other CPUs with AVX-512, then AVX2, then scalar code.

### Tests

```bash
pytest tests/                                     # C kernel vs Numba (skipped if not built), period validation
PYTFIT5_TEST_LC=lightcurve.dat pytest tests/      # also check a real light curve (time, flux columns)
```

## Quick start

```python
import pytfit5.transitPy5 as tpy5
import pytfit5.bls_cpu as gbls

phot = tpy5.readphot("lightcurve.dat")      # columns: time, flux [, ferr, itime]; first line is a header

inputs = tpy5.tpy5_inputs_class()           # one configuration object for detrending and search
inputs.mstar, inputs.rstar = 1.0, 1.0       # stellar mass and radius (solar units): set the duration grid
inputs.boxbin = 2.0                         # detrending window (days)
inputs.plots = 0                            # 0 = no plots

tpy5.run_cutoutliers(phot, inputs)          # flag outliers in phot.icut
tpy5.run_polyfilter_iterative(phot, inputs) # detrend into phot.flux_f, masking transit-like dips

keep = phot.icut == 0
ans = gbls.bls(inputs, phot.time[keep], phot.flux_f[keep])
print(f"P = {ans.bper:.5f} d, T0 = {ans.epo:.5f}, SNR = {ans.snr:.1f}, power = {ans.bpower:.1f}, "
      f"depth = {ans.depth*1e6:.0f} ppm, kernel = {ans.engine}")
```

With `multipro = 1` (the default), detrending and the Numba BLS kernel use a process pool. In a script
(not a notebook), put the code under `if __name__ == "__main__":`; Python 3.14 starts workers with
`forkserver` and fails without it.

See [`examples/roman_example.ipynb`](examples/roman_example.ipynb) for an end-to-end example on a
simulated Roman light curve (simulation, detrending, BLS search, checks, transit fit and DE-MCMC,
with the methods explained), and [API.md](API.md) for the full reference.

## Searches

| Function | What it finds | Returns |
|---|---|---|
| `bls_cpu.bls(inputs, time, flux)` | Periodic transits | `gbls_ans_class` |
| `bls_cpu.compute_pulse_search(time, flux, ...)` | The strongest single dip of 1-12 h (configurable) | `dict` (or `None`) |
| `bls_cpu.bls_pulse(inputs, time, flux)` | Runs both and keeps the one with the higher SNR | `gbls_ans_class` |

When `bls_pulse` keeps the single event, it sets `ans.single = True`, and `bper` is a period estimated
from the duration and the stellar density (always longer than the light curve). `bls` sets
`single = True` when its best period is longer than the light curve (possible only if `freq1` is
below 1/span).

### Settings that matter

| Setting | Default | Notes |
|---|---|---|
| `freq1`, `freq2` | auto: 2/span, 2 c/d | Period range is 1/`freq2` to 1/`freq1` days |
| `ofac` | 8 | Frequency oversampling |
| `nbins`, `max_bin_minutes` | 2000, 0 | Phase bins; with `max_bin_minutes > 0` long periods get more bins so no bin is wider than this |
| `dips_only` | `False` | Count only boxes below the median. `False` is the classic BLS statistic, which also finds brightenings (about half of the top peaks on pure noise) |
| `normalize` | `"iterative_baseline"` | Periodogram normalisation (see API.md) |
| `bls_engine`, `bls_isa` | `"auto"` | Kernel and SIMD variant |
| `multipro` | 1 | 1 = use all CPUs available to the process (respects Slurm/taskset), 0 = one thread |

For Roman Galactic Bulge Time Domain Survey light curves (12.1-min cadence, 70.5-day seasons) we use
`ofac = 4`, `nbins = 1500`, `max_bin_minutes = 5` and `dips_only = True`. In injection tests at a fixed
false-alarm rate, doubling `ofac` doubled the run time for at most 1.4 points of completeness.

### Detection statistics

`bls` reports two statistics for the best peak:

- `snr`: depth / scatter × √(points in transit), measured on the folded light curve. Use it to describe
  how strong a signal is.
- `bpower`: the height of the peak in the normalised periodogram. Use it to decide what counts as a detection.

The SNR at a given false-alarm rate rises with residual stellar variability and with the length of the
light curve; the normalised power changes much less. In simulated Roman light curves, `bpower ≥ 8.8`
(one season) or `≥ 9.3` (two seasons), together with `snr ≥ 6`, gave a false-alarm rate of 0.5-1% per
light curve for white noise and for detrended stellar variability alike. These thresholds depend on the
grid and the data, so calibrate them on noise-only light curves of your own survey.

## Detrending

`run_polyfilter(phot, inputs)` fits a polynomial of order `nfitp` in a moving window of `boxbin` days,
not across gaps longer than `gapsize`, and writes the detrended flux (same units as the input, median
preserved) to `phot.flux_f`. Points with `phot.tflag == 1` are excluded from the fits.

`run_polyfilter_iterative(phot, inputs)` repeats this, each time flagging transit-like dips in
`phot.tflag` so that they do not bias the trend:

- `iter_mask_mode = "points"`: runs of points below −`iter_sigma_threshold` × scatter, at least
  `iter_min_duration` days long;
- `iter_mask_mode = "binned"`: residuals averaged over `iter_bin_hours` below −`iter_sigma_threshold`
  × noise, which finds shallower and shorter dips.

`run_cutoutliers(phot, inputs)` flags outliers in `phot.icut`. `cut_direction = "up"` cuts only
upward outliers (flares, cosmic rays), so that transits lasting one or two cadences are kept.

The detrender and the outlier cut are compiled with Numba (`pytfit5/detrend.py`). One 70.5-day Roman
season (8,391 points) is detrended with both production windows (3 d and 1 d, cubic, binned masking)
in about 0.8 s on one core, against 56 s for the original Python version, with the same results to
~1e-10 of the noise. The original versions are kept as `polyfilter_reference` and
`cutoutliers_reference`. The first call in a new environment compiles the kernels (~7 s); they are
cached afterwards.

## Synthetic light curves

```python
from pytfit5.synthetic import generate_synthetic_lightcurve, compare_bls_injection

phot, sol = generate_synthetic_lightcurve(
    t0=5.0, per=10.0, time_length=60.0, depth=0.001, snr=12.0,
    cadence=1/96, stellar_noise_type="multi", stellar_noise_amplitude=0.002,
    rotation_period=12.0, seed=1)
```

Noise models: `"none"` (white only), `"red"` (1/f^α), `"drw"` (damped random walk), `"multi"`
(granulation + activity + spot modulation if `rotation_period` is given). `sol` is a
`transit_model_class` with the injected parameters; `compare_bls_injection(phot, sol, sol_bls)`
reports the overlap of injected and recovered transits, allowing for period aliases.

## Visualisation

```python
import pytfit5.transitplot as transitp
fig = transitp.plot_lightcurve_summary(phot, bls_ans=ans, tpy5_inputs=inputs)
```

Without `bls_ans` it shows the raw and detrended flux; with it, also the phased light curve and a
transit close-up with the BLS parameters.

## Contributing

Contributions are welcome. Open items:

1. Optimise GPU memory transfers (CUDA scripts).
2. Allow different transit-model parameters for different planets.
3. Better examples for TTV fitting.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

## Acknowledgments

Thank you to Canada Research Chairs, NSERC Discovery, Digital Research Alliance of Canada, Calcul Québec
and FRQNT for financial and hardware support.

This code was initially developed during the Bishop's University Winter Reading Week, making good use of
professional development resources.

The implementation of [TransitFit5](https://github.com/jasonfrowe/Kepler) in Python was developed by
Alexis Roy (Université de Sherbrooke), supported by an NSERC Undergraduate Student Research Award (USRA)
and an iREx Trottier Fellowship.

The BLS search follows Kovács, Zucker & Mazeh (2002), *A box-fitting algorithm in the search for periodic transits*.

If you find this code useful, please cite:
Rowe et al. 2014, ApJ, 784, 45;
Rowe et al. 2015, ApJS, 217, 16.

## Contact

Jason Rowe - jason@jasonrowe.org

## Change log

- 2026/11/19: Detrender (`polyfilter`) and outlier cut compiled with Numba: 60-100x faster, same
  results; `multipro=1` uses threads instead of a process pool. The outlier cut no longer uses stale
  samples for its scatter estimate in the first and last `nsampmax` points.
- 2026/11/19: `single` flag on search results; period validation fixed (epoch search, robust SNR,
  per-transit depths, SNR-gain threshold for aliases); README and API.md updated.
- 2026/11/18: C/OpenMP BLS kernel with run-time AVX-512/AVX2 selection (`bls_engine`, `bls_isa`);
  `dips_only` option.
- 2026/11/17: Fixed runaway masking in binned iterative detrending; NaN-safe MCMC run-away checks
  (`MCMCRunawayError`).
- 2026/10/31: Faster Numba BLS kernel; configurable phase bins (`nbins`, `max_bin_minutes`); robust
  depth and SNR in `bls_stats`; binned transit masking and upward-only outlier cuts in detrending.
- 2025/11/27: Big refresh of the code base. First steps to a pip-installable package.
- 2025/03/10: V2 is now faster on CPU.
- 2025/03/09: Added CPU version (Numba + threading); added 'V2' (best for TESS CVZ light curves; V1 best for Kepler).
- 2025/03/08: Initial version.
