# pytfit5 API Reference

**Version 1.0.0**

Reference for the `pytfit5` package (repository `bls_cuda`): Box Least Squares (BLS) transit search,
detrending, transit modelling and fitting, MCMC, synthetic light curves and period validation. For
installation and a quick start, see [README.md](README.md).

---

## Table of Contents

1. [Package Structure](#package-structure)
2. [Configuration: `tpy5_inputs_class`](#configuration-tpy5_inputs_class)
3. [Photometry and Detrending (`pytfit5.transitPy5`)](#photometry-and-detrending-pytfit5transitpy5)
4. [BLS Search (`pytfit5.bls_cpu`)](#bls-search-pytfit5bls_cpu)
5. [Compiled Kernel (`pytfit5.bls_c`)](#compiled-kernel-pytfit5bls_c)
   - [GPU Kernel (`pytfit5.bls_gpu`)](#gpu-kernel-pytfit5bls_gpu)
6. [Transit Model (`pytfit5.transitmodel`)](#transit-model-pytfit5transitmodel)
7. [Fitting and MCMC](#fitting-and-mcmc)
8. [Synthetic Light Curves (`pytfit5.synthetic`)](#synthetic-light-curves-pytfit5synthetic)
9. [Period Validation (`pytfit5.period_validation`)](#period-validation-pytfit5period_validation)
10. [Keplerian Utilities (`pytfit5.keplerian`)](#keplerian-utilities-pytfit5keplerian)
11. [Examples](#examples)
12. [Performance Notes](#performance-notes)

---

## Package Structure

Import the submodules directly; the package does not define short aliases.

```python
import pytfit5.bls_cpu as gbls            # BLS periodic and single-event searches
import pytfit5.transitPy5 as tpy5         # configuration, photometry I/O, detrending, MCMC I/O
from pytfit5 import bls_c                 # compiled C/OpenMP BLS kernel (optional extension)
from pytfit5 import bls_gpu               # CUDA BLS kernel (optional: pip install ".[cuda]")
import pytfit5.transitmodel as transitm   # transit model
import pytfit5.transitfit as transitf     # least-squares fitting
import pytfit5.transitmcmc as tmcmc       # DE-MCMC fitting
import pytfit5.mcmcroutines as mcmc       # MCMC samplers and diagnostics
import pytfit5.transitplot as transitp    # plots
import pytfit5.keplerian as kep           # orbital utilities
import pytfit5.synthetic as syn           # synthetic light curves, injection/recovery
from pytfit5 import period_validation as pval
```

Lower-level modules: `pytfit5.occult` (Mandel & Agol occultation functions) and `pytfit5.effects`
(albedo/phase modulation, TTV interpolation).

---

## Configuration: `tpy5_inputs_class`

`tpy5.tpy5_inputs_class()` holds all settings for photometry processing, detrending and the BLS
searches. (`gbls.gbls_inputs_class` is a deprecated alias of the same class.)

### Stellar parameters

| Parameter | Default | Description |
|---|---|---|
| `mstar`, `rstar` | `1.0`, `1.0` | Stellar mass and radius (solar units). Set the BLS duration range at each period |
| `teff`, `logg`, `feh` | `5777`, `4.5`, `0.0` | Effective temperature (K), log g (cgs), [Fe/H] |
| `e_mstar`, `e_rstar`, `e_teff`, `e_logg`, `e_feh` | `0.3`, `0.3`, `500`, `0.2`, `0.3` | Their uncertainties |

### BLS search

| Parameter | Default | Description |
|---|---|---|
| `freq1` | `-1` | Lowest frequency (c/d), i.e. longest period. `-1`: 2 / time span |
| `freq2` | `-1` | Highest frequency (c/d), i.e. shortest period. `-1`: 2 c/d |
| `ofac` | `8.0` | Frequency oversampling factor |
| `nbins` | `2000` | Number of phase bins (capped at npt − 1) |
| `max_bin_minutes` | `0.0` | If > 0, long periods get more bins, `max(nbins, P / max_bin_minutes)`, so no bin is wider than this (minutes). `0`: fixed `nbins` |
| `minbin` | `5` | Minimum number of points in a box |
| `dips_only` | `False` | Count only boxes whose flux is below the median (transits). `False` is the classic BLS statistic, which ranks brightenings equally; on pure noise about half the top peaks are then brightenings. The single-event search is always dips-only |
| `normalize` | `"iterative_baseline"` | Periodogram normalisation (see below) |
| `oneoverf_correction` | `True` | Subtract the running baseline of the periodogram before normalising |
| `oneoverf_extrapolate` | `True` | For periods longer than `oneoverf_threshold`, replace the baseline (and, in some modes, the noise) by its median over periods of 2/3 to 1 time span |
| `oneoverf_threshold` | `None` | Period (days) above which to extrapolate. `None`: the time span |
| `bls_engine` | `"auto"` | Kernel: `"auto"` (compiled C kernel if available, else Numba), `"c"` (error if unavailable), `"numba"`, `"cuda"` (GPU kernel, see [`bls_gpu`](#gpu-kernel-pytfit5bls_gpu); never chosen by `"auto"`). C and Numba results are bit-identical; the GPU's agree to ~1e-7 |
| `bls_isa` | `"auto"` | SIMD variant of the C kernel: `"auto"`, `"avx512"`, `"avx2"`, `"scalar"`, or explicitly `"avx512_nodiv"`, `"avx512_div"`, `"avx2_div"`, `"avx2_nodiv"` (see [`bls_c`](#compiled-kernel-pytfit5bls_c)) |
| `multipro` | `1` | `1`: use all CPUs available to the process (respects Slurm and taskset); `0`: one thread |
| `return_spectrum` | `False` | Store the periodogram in the result |
| `plots` | `1` | `0` none, `1` display, `2` PNG + display, `3` PNG only |
| `zerotime` | `0.0` | Subtracted from the reported epoch (e.g. a BJD offset) |
| `filename`, `lcdir` | `"filename.txt"`, `""` | Light curve read by `bls()` when no arrays are given; also the plot file name |
| `nper` | `50000` | Deprecated (the grid size is computed) |

### Single-event (pulse) search

| Parameter | Default | Description |
|---|---|---|
| `pulse_min_duration_hours` | `1.0` | Shortest event duration searched (hours) |
| `pulse_max_duration_hours` | `12.0` | Longest event duration searched (hours) |
| `pulse_bin_duration_days` | `None` | Binning before the search (days). `None`: 10 minutes |

### Detrending and outlier rejection

| Parameter | Default | Description |
|---|---|---|
| `boxbin` | `2.0` | Detrending window (days) |
| `nfitp` | `2` | Polynomial order (2 = quadratic) |
| `gapsize` | `0.5` | Gaps longer than this (days) are not detrended across |
| `iter_max_iters` | `5` | Maximum iterations of `run_polyfilter_iterative` |
| `iter_sigma_threshold` | `3.0` | Threshold for flagging a dip, in units of the scatter |
| `iter_min_duration` | `0.05` | `"points"` mode: shortest flagged dip (days) |
| `iter_mask_mode` | `"points"` | `"points"`: runs of points below the threshold; `"binned"`: residuals averaged over `iter_bin_hours` (finds shallower, shorter dips) |
| `iter_bin_hours` | `1.0` | Averaging window for `"binned"` mode (hours) |
| `iter_verbose` | `False` | Print progress |
| `cut_direction` | `"both"` | `run_cutoutliers`: `"both"` or `"up"` (only upward outliers, keeping 1-2 cadence transits) |
| `nsampmax`, `dsigclip` | `6`, `3.0` | Sample size and threshold of the derivative-based outlier cut |
| `sigclip`, `fstd_cut` | `3.0`, `5` | Simple sigma clipping |
| `boxwindow`, `boxsigma` | `2.0`, `3.0` | Window (days) and threshold for `time_window_clip` |

---

## Photometry and Detrending (`pytfit5.transitPy5`)

### `phot_class`

Container for a light curve. Arrays: `time`, `flux`, `ferr`, `itime` (integration time per point,
days), `tflag` (1 = in transit, excluded from detrending fits), `icut` (1 = rejected point), `flux_f`
(detrended flux).

### `readphot(filename)`

Read an ASCII light curve: a header line, then columns time, flux, and optionally ferr and itime.
Missing `ferr` is set to the standard deviation of the flux; missing `itime` to the median cadence.
Returns a `phot_class` (`flux_f` empty until detrending).

### `run_cutoutliers(phot, inputs)`

Flag outliers in `phot.icut` (derivative-based cut with `nsampmax` and `dsigclip`, direction
`cut_direction`). A point is cut if it differs from both neighbours, in the same direction, by more
than `dsigclip` × the median absolute point-to-point difference within ±`nsampmax` points. Uses
`cutoutliers(time, flux, nsampmax, sigma, direction)` (compiled; data in time order);
`cutoutliers_reference` is the original version, which used stale samples for the scatter in the
first and last `nsampmax` points.

### `run_polyfilter(phot, inputs)`

Polynomial detrending (order `nfitp`, window `boxbin` days, not across gaps > `gapsize`), ignoring
points with `tflag == 1`. Writes `phot.flux_f` = flux − trend + median(flux), i.e. in the units of
the input. Calls `polyfilter`.

### `polyfilter(time, flux, ferr, tflag, boxbin, nfitp, gapsize, multipro=1)`

The detrender (compiled with Numba, `pytfit5/detrend.py`). At each point, a polynomial of order
`nfitp` is fitted, with weights 1/`ferr`², to the unmasked points within ±`boxbin`/2 that are not
across a gap longer than `gapsize`, with iterative 4σ clipping (up to 10 iterations), and evaluated
at that point. For a masked point, the fit needs at least 2(`nfitp`+1) unmasked points on both sides;
otherwise the window is widened ×2 and ×4, then a straight line across the masked stretch, the
median of the nearest points (one side only), or the median of the window is used. `time` need not
be sorted. `multipro=1` runs on Numba threads (`NUMBA_NUM_THREADS`), `0` on one thread. Returns
flux − trend + median(flux). `polyfilter_reference` is the original pure-Python version (same
results, 60-100× slower).

### `run_polyfilter_iterative(phot, inputs)`

Repeats `run_polyfilter`, each time flagging transit-like dips in `phot.tflag` (mode
`iter_mask_mode`), until the mask converges or `iter_max_iters` is reached. Modifies `phot.flux_f`
and `phot.tflag` in place.

### `save_mcmc_results(filename, phot, sol, chain, accept, burnin, params_to_fit, x=None, beta=None, serr=None, params=None)` / `load_mcmc_results(filename)`

Save or load an MCMC run (photometry, solution, chain, acceptance, burn-in, fitted parameter names,
optional MCMC inputs) in HDF5. `load_mcmc_results` returns
`phot, sol, chain, accept, burnin, params_to_fit, metadata`. Requires `h5py`.

---

## BLS Search (`pytfit5.bls_cpu`)

### `gbls_ans_class`

Result of a search.

| Attribute | Description |
|---|---|
| `bper` | Period of the best peak (days). From `bls_pulse` when the single event wins: a period estimated from the duration and stellar density (see below) |
| `epo` | Epoch (mid-transit time) in the input time system minus `zerotime` |
| `tdur` | Duration (days) |
| `depth` | Depth: median out-of-transit flux − mean in-transit flux (positive for a dip) |
| `snr` | depth / σ × √(points in transit), σ = 1.4826 × MAD of the out-of-transit flux |
| `bpower` | Height of the best peak in the normalised periodogram (for `bls_pulse` single events: the pulse SNR) |
| `engine` | Kernel used: `"numba"` or `"c:<variant>"` |
| `single` | `True` for a single event: from `bls_pulse` when the pulse search wins, from `bls` when the best period is longer than the time span |
| `periods`, `power`, `freqs` | Periodogram, if `return_spectrum=True` (else `None`) |

**Detection statistics.** `snr` measures how strong a signal is; `bpower` is the better statistic for
deciding what is a detection. The SNR needed for a given false-alarm rate rises with residual stellar
variability and with the length of the light curve, while the normalised power changes much less. On
simulated Roman light curves (`ofac=4`, `nbins=1500`, `max_bin_minutes=5`, `dips_only=True`),
`bpower ≥ 8.8` (one 70.5-d season) or `≥ 9.3` (two seasons) together with `snr ≥ 6` gave a false-alarm
rate of 0.5-1% per light curve for white noise and for detrended stellar variability. Calibrate the
thresholds on noise-only light curves of your own data.

### `bls(inputs, time, flux)`

Periodic BLS search.

- `inputs` (`tpy5_inputs_class`); `time`, `flux` (`np.ndarray`, flux around 1). If the arrays have fewer
  than two points, the file `inputs.lcdir/inputs.filename` is read with `readfile`.
- Internally time starts at 0 and the median flux is subtracted; the inputs are not modified.
- Returns a `gbls_ans_class`.

Steps: the frequency grid runs from `freq1` to `freq2` with a step proportional to the expected
duration at each frequency (from `mstar`, `rstar`) and inversely to `ofac`; each trial period is folded
into `nbins` phase bins (more at long periods with `max_bin_minutes`), and the best box over durations
compatible with the star is found; the periodogram is normalised, its highest peak gives the period,
and depth and SNR are measured on the light curve folded at that period. Periods longer than the time
span (only if `freq1` < 1/span) use a single-event kernel.

```python
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5

inputs = tpy5.tpy5_inputs_class()
inputs.freq1 = 1.0 / 20.0   # longest period 20 d
inputs.freq2 = 2.0          # shortest period 0.5 d
inputs.plots = 0
ans = gbls.bls(inputs, time, flux)
print(f"P = {ans.bper:.6f} d, T0 = {ans.epo:.6f}, SNR = {ans.snr:.2f}, power = {ans.bpower:.2f}")
```

### Periodogram normalisation (`normalize`)

All modes work on √(BLS power):

- `"iterative_baseline"` (**default, recommended**): a running median baseline is computed with strong
  peaks iteratively masked (3σ, up to 3 iterations); with `oneoverf_correction=True` it is subtracted;
  the result is divided by one global noise level (1.4826 × MAD of the unmasked points).
- `"mad"`: running median baseline and running MAD noise.
- `"percentile_mad"`: running 75th-percentile baseline and MAD noise.
- `"coverage_mad"`: MAD noise weighted by the expected duty cycle at each frequency (also used for
  an unrecognised mode name).
- `"none"`: √power, baseline-subtracted if `oneoverf_correction=True`.

The running-window width follows the expected transit duty cycle. With `oneoverf_extrapolate=True`,
the baseline at periods longer than `oneoverf_threshold` (default: the time span) is set to its median
over periods of 2/3 to 1 time span, so that long periods are not suppressed. In our tests on simulated
Roman light curves the alternatives to `"iterative_baseline"` put the top peak on the injected period
far less often.

### `compute_pulse_search(time, flux, min_duration_hours=1.0, max_duration_hours=12.0, bin_duration_days=None)`

Single-event search: the strongest dip of any duration in the range, anywhere in the light curve.

- `time` must be sorted. The median flux is subtracted internally, so no preprocessing is needed.
- The flux is binned (`bin_duration_days`, default 10 min); every window of every width is scored as
  |Σ flux| / (σ √n), with σ = 1.4826 × MAD of the flux; only dips (Σ flux < 0) count.
- Returns `None` if no durations fit, else a dict:
  - `'t0'`: centre of the event (same time system as `time`)
  - `'duration'`: days
  - `'depth'`: mean flux in the window relative to the median (negative for a dip)
  - `'snr'`: as above
  - `'spectrum_duration'` (hours), `'spectrum_snr'`: best SNR for each duration

The SNR is the maximum over many correlated trials and is not corrected for their number, so set a
threshold from noise-only light curves.

```python
res = gbls.compute_pulse_search(phot.time, phot.flux_f, min_duration_hours=1.0, max_duration_hours=12.0)
if res is not None:
    print(f"t0 = {res['t0']:.4f}, duration = {res['duration']*24:.2f} h, SNR = {res['snr']:.1f}")
```

### `bls_pulse(inputs, time, flux)`

Runs `bls()` and `compute_pulse_search()` on the same data and returns the result with the higher SNR
as a `gbls_ans_class`.

- If the periodic search wins: the `bls()` result.
- If the single event wins: `single = True`, `epo` = event time (minus `zerotime`), `tdur`, `depth`
  (positive for a dip), `snr` = `bpower` = pulse SNR, and `bper` = the period from
  `period_validation.estimate_single_transit_period` (duration + stellar density from `mstar`,
  `rstar`), which is always at least the time span + 1.5 × duration.

Plots (`plots > 0`) show the periodogram or the single event, whichever won.

```python
inputs.pulse_min_duration_hours = 1.0
inputs.pulse_max_duration_hours = 8.0
ans = gbls.bls_pulse(inputs, time, flux)
if ans.single:
    print(f"single event at {ans.epo:.4f}, duration {ans.tdur*24:.1f} h, SNR {ans.snr:.1f}")
else:
    print(f"periodic: P = {ans.bper:.5f} d, SNR {ans.snr:.1f}")
```

| | `bls()` | `compute_pulse_search()` | `bls_pulse()` |
|---|---|---|---|
| Finds periodic signals | yes | no | yes |
| Finds single events | only if `freq1` < 1/span | yes | yes |
| Cost | full search | small fraction of `bls()` | both |
| Returns | `gbls_ans_class` | `dict` | `gbls_ans_class` |

### `readfile(filename)`

Read a whitespace-separated file with one header line; returns `time, flux` from the first two columns.

### `calc_eph(p, jn1, jn2, npt, time, flux, freqs, ofac, nstep, nb, mintime, Keptime, Mstar, Rstar, normalize_mode="iterative_baseline", oneoverf_correction=False, oneoverf_extrapolate=True, oneoverf_threshold=None)`

Internal: normalises the raw periodogram `p` and derives `periods, power, bper, epo, bpower, snr, tdur,
depth` for the best peak. Called by `bls()`; note that the defaults of its keyword arguments differ from
those of `tpy5_inputs_class`.

### Constants

`G` = 6.674×10⁻¹¹ N m² kg⁻², `Rsun` = 6.96265×10⁸ m, `Msun` = 1.9891×10³⁰ kg,
`pifac` = (2π)^(2/3)/π, `day2sec` = 86400, `onehour` = 1/24 d.

---

## Compiled Kernel (`pytfit5.bls_c`)

ctypes interface to the optional C/OpenMP BLS kernel (`pytfit5/_cbls.c`, built by `pip install`).
`bls()` uses it automatically; these functions are for checking the build and for direct use.

| Function | Description |
|---|---|
| `available()` | `True` if the compiled kernel loaded |
| `load_error()` | Why it did not load (`None` if it did) |
| `has_openmp()` | `True` if built with OpenMP. Without it, `bls_engine="auto"` with `multipro=1` uses Numba |
| `supported_variants()` | Variants this CPU can run |
| `cpu_vendor()` | `"AuthenticAMD"`, `"GenuineIntel"`, ... (from `/proc/cpuinfo`; `""` if unknown) |
| `resolve_variant(isa="auto")` | The concrete variant an `isa` request maps to (raises if unsupported) |
| `scan(freqs, nbs, time, flux, Rstar=1.0, Mstar=1.0, isa="auto", nthreads=0, dips_only=False)` | Raw BLS over all frequencies; returns `p, jn1, jn2` bit-identical to the Numba kernel. `time` must start at 0 and `flux` be median-subtracted; periods must not exceed the time span |

Variants: `scalar`, `avx2_div`, `avx2_nodiv`, `avx512_div`, `avx512_nodiv` (aliases: `avx512` →
`avx512_nodiv`, `avx2` → `avx2_div`). `"auto"` picks the first supported of `avx512_div`, `avx2_div`,
`scalar` on AMD CPUs, and of `avx512_nodiv`, `avx2_div`, `scalar` otherwise. On an AMD EPYC 9654 (Zen 4)
`avx512_div` was 12% faster than `avx512_nodiv`; on an Intel Xeon Silver 4116 (Skylake-SP) the
division-free variant was faster.

### GPU Kernel (`pytfit5.bls_gpu`)

CUDA BLS kernel for NVIDIA GPUs, written with `numba-cuda` (install with `pip install ".[cuda]"`).
`bls()` uses it with `bls_engine="cuda"`; frequencies with periods longer than the time span still use
the Numba single-event kernel, as with the C kernel.

| Function | Description |
|---|---|
| `available()` | `True` if `numba-cuda` imported and a GPU is visible |
| `load_error()` | Why it is unavailable (`None` if it is available) |
| `max_nb()` | Largest number of phase bins searched in shared memory (8 bytes per bin; ~12,000 with 100 KB per block) |
| `scan(freqs, nbs, time, flux, Rstar=1.0, Mstar=1.0, dips_only=False, global_mem_mb=512, max_shared_nb=None)` | Same inputs and outputs as `bls_c.scan`. Frequencies with `nb > max_nb()` keep their bins in global memory, in launches of at most `global_mem_mb` MB |
| `pack_flux(flux)` | The fixed-point packing used for binning (see below); returns `packed, cb, 2^-e` |
| `limits(freqs, nbs, npt, Rstar, Mstar)` | Box-width limits `kmi, kma, kkmi` per frequency, as in the CPU kernels |

How it differs from the CPU kernels. One thread block searches one frequency.

- **Phase**: FP64, the CPU kernels' operations, so every point falls in the same bin.
- **Binning**: each point is one int64, `round(flux 2^e) << cb` plus 1, so one shared-memory integer
  atomic adds both its flux and its count. Integer sums are exact and independent of the order in
  which threads add them, so results are deterministic.
- **Search**: FP32 running sums with the division-free comparison of the C kernel
  (`s² D_best > N_best dfac`), ties going to the first start bin as on the CPU. FP64 runs at 1/32-1/64
  of the FP32 rate on most NVIDIA cards (1/31 measured on an RTX PRO 4000 Blackwell).
- **Power**: the winning box's power is recomputed in FP64 from its FP32 bin sums.

So `p` agrees with the CPU to ~1e-7 relative and `jn1, jn2` agree except where two boxes have the same
power to FP32 precision (none in the tests).

---

## Transit Model (`pytfit5.transitmodel`)

### `transit_model_class`

Model parameters. Per-planet parameters are lists with one entry per planet.

| Parameter | Default | Description |
|---|---|---|
| `rho` | `1.0` | Mean stellar density (g/cm³) |
| `nl1`, `nl2` | `0.0` | Non-linear limb-darkening coefficients (unused for quadratic) |
| `nl3`, `nl4` | `0.3`, `0.4` | Quadratic limb darkening in the Kipping (2013) parametrisation, q1 and q2 |
| `dil` | `0.0` | Dilution |
| `vof` | `0.0` | RV offset |
| `zpt` | `0.0` | Photometric zero point |
| `dscale` | `1.0` | Error-bar scale in the MCMC likelihood |
| `npl` | `1` | Number of planets |
| `t0` | `[0.0]` | Mid-transit time (days) |
| `per` | `[1.0]` | Period (days) |
| `bb` | `[0.5]` | Impact parameter |
| `rdr` | `[0.1]` | Rp/R* |
| `ecw`, `esw` | `[0.0]` | √e cos ω, √e sin ω, with ω the argument of periastron as in the radial-velocity literature (transit at true anomaly f = π/2 − ω). In pytfit5 before Nov 2026 ω was the true anomaly at mid-transit; convert old values with `keplerian.from_legacy_transitfit5`. |
| `krv` | `[0.0]` | RV semi-amplitude (m/s) |
| `ted` | `[0.0]` | Thermal eclipse depth (ppm) |
| `ell` | `[0.0]` | Ellipsoidal amplitude (ppm) |
| `alb` | `[0.0]` | Albedo amplitude (ppm) |

Each parameter `x` has an uncertainty `dx`. Methods: `to_array()`, `from_array(arr)`,
`err_to_array()`, `load_errors(err_arr)`.

### `transitModel(sol, time, itime=-1, nintg=41, ntt=-1, tobs=-1, omc=-1)`

Model light curve.

- `sol`: `transit_model_class` or the array from `to_array()`.
- `time`: times (days); `itime`: integration time per point (days; default 30 min).
- `nintg`: sub-samples per integration.
- `ntt`, `tobs`, `omc`: transit-timing variations: number per planet, shape `(npl,)`; their times and
  O−C values, shape `(npl, nmax)`.
- Returns the model flux (same length as `time`).

```python
import numpy as np
import pytfit5.transitmodel as transitm

sol = transitm.transit_model_class()
sol.t0, sol.per, sol.bb, sol.rdr = [5.0], [3.5], [0.3], [0.1]
sol.rho = 1.4
time = np.linspace(0.0, 10.0, 5000)
itime = np.full_like(time, 30.0 / 1440.0)
model = transitm.transitModel(sol, time, itime, nintg=41)
```

---

## Fitting and MCMC

### `transitfit.fitTransitModel(sol_obj, params_to_fit, phot, nintg=41, ntt=-1, tobs=-1, omc=-1, **ls_kwargs)`

Least-squares fit of the parameters named in `params_to_fit` (e.g.
`["rho", "zpt", "t0", "per", "bb", "rdr"]`) to `phot`. It uses `flux_f`, `ferr` and `itime` of the
points with `icut == 0` **and** `tflag == 1`, so mark the data around the transits first with
`keplerian.mark_intransit_data(phot, sol)` (note that `run_polyfilter_iterative` also writes
`tflag`). Returns a new `transit_model_class` with the fitted values and uncertainties.
`demcmcRoutine` selects its data the same way.

Extra keyword arguments (`ls_kwargs`) are passed to `scipy.optimize.least_squares`. Its default
stopping tolerances (1e-8) can leave a high-SNR fit (e.g. Kepler KOI-1) short of the minimum by more
than the formal errors; for a tightly converged fit use, for example,
`fitTransitModel(sol, params_to_fit, phot, ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=2000)`.

### DE-MCMC (`pytfit5.transitmcmc`)

| Function | Description |
|---|---|
| `genmcmcInput(sol, params_to_fit, priors=None)` | Returns `lnprob, x, beta`: the log-probability function, starting vector and initial step sizes. `priors`: optional `priors_class` with Gaussian priors |
| `cutOutOfTransit(sol, phot, tdurcut=2)` | Keep only data within `tdurcut` durations of the transits |
| `demcmcRoutine(x, beta, phot, sol_a, serr, params, lnprob, nintg=41, ntt=-1, tobs=-1, omc=-1)` | Run DE-MCMC; returns `chain, accept, burnin`. Raises `MCMCRunawayError` if the chain runs away |
| `getParams(chain, burnin, sol, params_to_fit)` | Posterior summary as a `transit_model_class` |

`params = [nsteps1, nsteps2, nsteps_inc, burninf, niter_cor, burnin_cor, nthin, nloopmax,
converge_crit, buf_converge_crit]`: test-chain length, main-chain length, increment per loop, burn-in
fraction, steps and burn-in for the step-size correlation, thinning, maximum loops, and the
Gelman-Rubin convergence criteria for the chain and the buffer. The example notebook uses
`[10000, 50000, 50000, 0.2, 10000, 1000, 101, 5, 1.02, 1.2]`.

```python
import pytfit5.transitfit as transitf
import pytfit5.transitmcmc as tmcmc
import pytfit5.transitplot as transitp

import pytfit5.keplerian as kep

params_to_fit = ["rho", "zpt", "t0", "per", "bb", "rdr"]
kep.mark_intransit_data(phot, sol)          # tflag = 1 within ~2 durations of each transit
sol_fit = transitf.fitTransitModel(sol, params_to_fit, phot)
lnprob, x, beta = tmcmc.genmcmcInput(sol_fit, params_to_fit)
phot_cut = tmcmc.cutOutOfTransit(sol_fit, phot)
params = [10000, 50000, 50000, 0.2, 10000, 1000, 101, 5, 1.02, 1.2]
chain, accept, burnin = tmcmc.demcmcRoutine(x, beta, phot_cut, sol_fit.to_array(),
                                            sol_fit.err_to_array(), params, lnprob)
sol_mcmc = tmcmc.getParams(chain, burnin, sol_fit, params_to_fit)
transitp.printParams(sol_mcmc)
transitp.plotTransit(phot, sol_mcmc, pl_to_plot=1)
tpy5.save_mcmc_results("mcmc.h5", phot_cut, sol_mcmc, chain, accept, burnin, params_to_fit,
                       x=x, beta=beta, params=params)
```

Samplers and diagnostics (`mcmcroutines`): `mhgmcmc`, `demhmcmc`, `genchain`, `gelmanrubin`,
`calcacrate`, `betarescale`, and plots `plotchains`, `triplot`, `plotmodels`.

### Plots (`pytfit5.transitplot`)

- `plot_lightcurve_summary(phot, bls_ans=None, tpy5_inputs=None, figsize=(14, 10))`: raw and detrended
  flux; with `bls_ans`, also the phased light curve and a transit close-up. Returns the figure.
- `plotTransit(phot, sol, pl_to_plot=1, ..., filename=None, y1=None, y2=None, xdur_scale=1.5)`: phased
  data and model for one planet.
- `plotTransitMCMC(chain, burnin, sol, params_to_fit, phot, ...)`, `printParams(sol)`.

---

## Synthetic Light Curves (`pytfit5.synthetic`)

### `generate_synthetic_lightcurve(t0, per, time_length, depth, snr, cadence=1/48, stellar_noise_type='multi', stellar_noise_amplitude=0.003, rotation_period=None, seed=None, nintg=41, itime=None)`

One planet (circular orbit) modelled with `transitModel`, plus white and stellar noise.

- `t0`, `per`, `time_length` (days); `depth` (fractional); `cadence` (days).
- `itime`: exposure time per point (days). The default (`None`) is the cadence, i.e. no dead time;
  set it when the exposure is shorter than the cadence, e.g. Roman: `cadence=12.1/1440`,
  `itime=67/86400`.
- `nintg`: sub-exposures used to average the model over each exposure (default 41). Use a large value
  (e.g. 1001) for a "true" light curve against which fits with fewer sub-exposures are tested.
- `snr`: the white noise σ is set so that Σ(in-transit signal) / (σ √n_in) = `snr`.
- `stellar_noise_type`: `"none"`, `"red"` (1/f^α), `"drw"` (damped random walk), `"multi"`
  (granulation + activity DRWs, plus spot modulation if `rotation_period` is given), with RMS amplitude
  `stellar_noise_amplitude`.
- Returns `phot, sol`: a `phot_class` (time starting at 0, flux around 1, `ferr` = σ, `flux_f` = copy
  of `flux`) and the injected `transit_model_class`.

Noise generators: `generate_red_noise(time, amplitude, alpha=1.0, seed=None)`,
`generate_drw_noise(time, amplitude, tau, seed=None)`,
`generate_stellar_noise(time, granulation_amp=0.001, activity_amp=0.002, rotation_period=None, spot_amp=0.005, seed=None)`.

### `compare_bls_injection(phot, sol_injected, sol_bls, verbose=True)`

Compares recovered and injected transits (wrapper around `calculate_transit_overlap`). Returns a dict
with `overlap_fraction`, `precision`, `true_positive`, `false_positive`, `false_negative`,
`period_factor`, `period_factor_type` (`'exact'`, `'multiple'`, `'fraction'`, `'mismatch'`) and
`is_recovered` (overlap and precision both > 50%).

### `calculate_transit_overlap(time, true_t0, true_per, true_duration, recovered_t0, recovered_per, recovered_duration, max_period_factor=5)`

Same statistics from explicit parameters, testing period aliases up to `max_period_factor`.

### `mark_in_transit(time, t0, per, duration)`

Boolean array of in-transit points.

---

## Period Validation (`pytfit5.period_validation`)

### `validate_bls_period(time, flux, bls_result, test_factors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0], optimize_t0=True, verbose=True, analyze_transits=True, min_snr_ratio=1.10)`

Tests multiples and fractions of the BLS period and individual transits. `flux` must be centred on 0.

- Each trial period is scored with `compute_transit_snr` (below). For a factor k ≥ 1, the k possible
  epochs t0 + j·P (j = 0 … k−1) are each refined within ±1 duration; for k < 1, t0 is refined.
- Another factor replaces the BLS period only if its SNR exceeds the SNR at factor 1 by
  `min_snr_ratio`. A true P/2 or 2P alias gains about √2; on simulated 60-day light curves with SNR
  6-10, 1.10 changed 1.7% of correct periods and missed 3.3% of aliases.
- A transit is "single dominant" if the other transits folded together have less than a third of the
  SNR of the strongest one.

Returns `results, transit_info`:

- `results`: `best_period`, `best_t0`, `best_factor`, `best_snr`, `best_depth`, `best_n_transits`,
  `is_alias`, `is_single_transit`, `snr_improvement`, and per-factor arrays `factors`, `periods`,
  `t0s`, `snrs`, `depths`, `n_transits`.
- `transit_info` (`None` unless `analyze_transits` and more than one transit): `transit_numbers`,
  `transit_times`, `transit_snrs`, `transit_depths`, `strongest_snr`, `rest_snr` (all other transits
  folded together), `mean_snr`, `median_snr`, `std_snr`, `is_single_dominant`, `n_transits`.
  If `is_single_dominant`, `results['is_single_transit']` is set to `True`.

### `estimate_single_transit_period(time, tdur, rho_star=1.41, margin_factor=1.5)`

Period for a single transit, assuming b = 0 and e = 0:
P_ρ = π² G ρ* τ³ / 3. If P_ρ would put a second transit in the data, the minimum period
P_min = time span + `margin_factor` × τ is returned instead. Returns
`period_estimate, period_min, is_constrained, n_transits_expected`.

### Other functions

- `compute_transit_snr(time, flux, t0, period, tdur, n_dur=1.5)` → `snr, depth, n_transits, in_transit_mask`.
  In transit: within `tdur/2` of mid-transit; out of transit: beyond `n_dur × tdur/2`. depth = median
  out-of-transit flux − mean in-transit flux; SNR = depth / σ × √n_in with σ = 1.4826 × MAD of the
  out-of-transit flux.
- `analyze_individual_transits(time, flux, t0, period, tdur, n_dur=1.5)` → the `transit_info` dict
- `optimize_epoch(time, flux, period, tdur, t0_initial, search_window=None)` → `t0, snr, depth, n_transits`.
  Grid over `search_window` (default `period/2`) in steps of at most `tdur/10`, always including `t0_initial`.
- `test_period_aliases(time, flux, t0_bls, period_bls, tdur_bls, test_factors=..., optimize_t0=True, min_snr_ratio=1.10)` → `results`
- `get_stellar_density(mass_solar, radius_solar)` → mean density (g/cm³)

---

## Keplerian Utilities (`pytfit5.keplerian`)

- `rhostar(P, tdur)`: stellar density (g/cm³) from period and duration (days), assuming b = 0, e = 0
  and a/R* ≫ 1: ρ* = 3P / (π² G τ³).
- `transitDuration(sol, i_planet=0)`: total duration T14 (days) of planet `i_planet`, including
  eccentricity (Kipping 2010; same as `transitdur.f` of TransitFit5).
- `ecc_omega(ecw, esw)`, `ecw_esw(e, omega)`: convert between the fitted parameters and e, ω (radians).
- `radial_velocity(time, t0, per, K, ecw, esw, gamma=0)`: the star's radial velocity
  K[cos(f + ω) + e cos ω] + γ (positive away from the observer) for a planet with mid-transit time `t0`.
- `ecc_params_from_state(r, v, mu, observer=(0, 0, 1))`: `ecw, esw, e, omega` from the planet's position
  and velocity relative to the star (e.g. from an N-body code), with `mu` = G(M* + Mp) and `observer` the
  direction from the system to the observer: e sin ω = **e**·ô and e cos ω = **e**·(ô × ĥ), with ô the
  observer direction projected onto the orbit plane.
- `from_legacy_transitfit5(ecw_old, esw_old)`: convert parameters from pytfit5 before Nov 2026 or TransitFit5
  between 14 Aug and Nov 2026, when ω was the true anomaly at mid-transit (the conversion swaps the two).
- `mark_intransit_data(phot, sol, tdurcut=2.0)`: reset `phot.tflag` and set it to 1 for points within
  about `tdurcut` durations of each transit of every planet in `sol` (the data used by the fits).
- `solve_kepler_eq(eccn, Manom, Eanom, thres=1e-6, itmax=100)`, `trueAnomaly(eccn, Eanom)`,
  `distance(a, eccn, Tanom)`.

---

## Examples

In scripts (not notebooks), run these under `if __name__ == "__main__":`; see the README.

### Search and validate

```python
import numpy as np
import pytfit5.bls_cpu as gbls
import pytfit5.transitPy5 as tpy5
from pytfit5 import period_validation as pval

phot = tpy5.readphot("star_lightcurve.txt")
inputs = tpy5.tpy5_inputs_class()
inputs.mstar, inputs.rstar = 0.8, 0.75
inputs.plots = 0
tpy5.run_cutoutliers(phot, inputs)
tpy5.run_polyfilter_iterative(phot, inputs)
keep = phot.icut == 0
time, flux = phot.time[keep], phot.flux_f[keep]

ans = gbls.bls(inputs, time, flux)
print(f"P = {ans.bper:.4f} d, T0 = {ans.epo:.4f}, SNR = {ans.snr:.1f}, power = {ans.bpower:.1f}")

results, transit_info = pval.validate_bls_period(time, flux - np.median(flux), ans, verbose=True)
if results['is_alias']:
    ans.bper, ans.epo = results['best_period'], results['best_t0']
elif results['is_single_transit']:
    rho = pval.get_stellar_density(inputs.mstar, inputs.rstar)
    ans.bper, *_ = pval.estimate_single_transit_period(time, ans.tdur, rho_star=rho)
```

### Injection test with the periodogram

```python
import matplotlib.pyplot as plt
from pytfit5.synthetic import generate_synthetic_lightcurve

phot, sol_injected = generate_synthetic_lightcurve(
    t0=2.5, per=8.3, time_length=100.0, depth=0.002, snr=20.0,
    stellar_noise_type="none", seed=123)

inputs = tpy5.tpy5_inputs_class()
inputs.plots = 0
inputs.return_spectrum = True
ans = gbls.bls(inputs, phot.time, phot.flux)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(phot.time, phot.flux, 'k.', ms=2)
a1.set_xlabel('Time (d)'); a1.set_ylabel('Flux')
a2.plot(ans.periods, ans.power, 'b-', lw=0.5)
a2.axvline(ans.bper, color='r', ls='--', label=f'Detected: {ans.bper:.2f} d')
a2.axvline(8.3, color='g', ls=':', label='Injected: 8.30 d')
a2.set_xlabel('Period (d)'); a2.set_ylabel('Normalised power'); a2.legend()
plt.show()
```

### Single transits

```python
phot, sol_injected = generate_synthetic_lightcurve(
    t0=10.0, per=85.0, time_length=60.0, depth=0.008, snr=12.0, seed=456)

inputs = tpy5.tpy5_inputs_class()
inputs.plots = 0
ans = gbls.bls_pulse(inputs, phot.time, phot.flux)
if ans.single:
    print(f"Single event at {ans.epo:.3f} d, duration {ans.tdur*24:.1f} h, SNR {ans.snr:.1f}; "
          f"period estimate {ans.bper:.0f} d (injected {sol_injected.per[0]:.0f} d)")
else:
    print(f"Periodic peak at {ans.bper:.2f} d, SNR {ans.snr:.1f}")

res = gbls.compute_pulse_search(phot.time, phot.flux)
print(f"Pulse search: t0 = {res['t0']:.3f} d, duration {res['duration']*24:.1f} h, SNR {res['snr']:.1f}")
```

A single deep transit also makes a periodic peak, at a period that leaves the other transits in gaps
or in the noise. In this example (seed 456) the periodic search wins with P = 23 d and SNR 8.2, so
`bls_pulse` returns it; the pulse search on its own finds the transit at t = 10 d. Compare both
results, or check the number of transits the periodic solution actually covers, before accepting a
long period.

---

## Performance Notes

- **Kernels.** The compiled C kernel (OpenMP threads, AVX-512/AVX2) is used when available and is
  1.9-4× faster than the Numba kernel (process pool) for a full search, with identical results. On one
  core of an AMD EPYC 9654, a 70.5-day Roman season (8,391 points) takes about 2.6 s with the C kernel
  and 4.8 s with Numba (`ofac=4`, `nbins=1500`, `max_bin_minutes=5`).
- **Threads.** `multipro=1` uses the CPUs available to the process (Slurm allocation or taskset), not
  the whole node. For many light curves, it is usually faster to run one light curve per core with
  `multipro=0`.
- **Grid.** Run time scales roughly with `ofac` and with the number of phase bins.
- **Detrending.** Compiled: about 0.8 s per Roman season for both production windows on one core
  (was 56 s), so it is now a few percent of the pipeline time instead of a third to a half.
- **First call.** Numba compiles its kernels on first use and caches them (`cache=True`), so the first
  search and detrending in a new environment are slower (detrender: ~7 s).

---

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later).

## Citation

If you use this software, please cite Rowe et al. 2014, ApJ, 784, 45 and Rowe et al. 2015, ApJS, 217, 16,
and the repository:

```
Rowe, J. F. et al., pytfit5: Python implementation of TransitFit5 and BLS.
https://github.com/jasonfrowe/bls_cuda
```

## Support

- Issues: https://github.com/jasonfrowe/bls_cuda/issues
- Interactive example: `examples/roman_example.ipynb`
- Contact: jason@jasonrowe.org
