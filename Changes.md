# BLS CUDA Optimization Changes

### Date: 2026-03-04
### Target: `bls_cuda_v2.py` / `bls_cuda_v2.ipynb`

## 1. Resolved CUDA Environment Architectural Error (`compute_61`)
* **Issue**: The system CUDA Toolkit was upgraded to 13.1, which dropped support for the Pascal architecture (`compute_61`) used by the GTX 1080 Ti.
* **Fix**: Installed PyPI/pip versions of the `cuda-toolkit` 12.x wheels in the virtual environment. Included a dynamic bootstrap script at the top of `bls_cuda_v2.py` to intercept the Numba environment lookup and forcefully point it to `cuda_home` symlinks populated with the CUDA 12 packages. This natively restored full support for the GTX 1080 Ti without modifying system drivers.

## 2. Kernel Bottleneck Optimization Phase 1
* **Issue**: The kernel was performing `cuda.atomic` operations in the innermost loop over global memory arrays (`power_g`, `jn1_g`, `jn2_g`). Global memory access is slow, and hitting it thousands of times per thread sequentially bottlenecked execution.
* **Fix**: Introduced `best_power`, `best_i`, `best_j` variables locally within the GPU registers for each thread's frequency calculation. The thread evaluates the max power entirely within fast registers and writes replacing the global memory atomic calls once per iteration at the end.
* **Result**: Reduced evaluation time by ~15% for the benchmark sweep (`7.93s -> 7.29s`).

## 3. Kernel Bottleneck Optimization Phase 2: Block Collaboration
* **Issue**: The kernel mapped 1 Thread to 1 Frequency. This meant all 128 threads simultaneously attempted to bin 128 different frequencies into thread-local arrays, spilling `3 MB` of local arrays into slow global VRAM continuously.
* **Fix**: Re-architected `compute_bls` and `bls_kernel` to launch **1 Block per Frequency**. All 128 threads per block now share the exact same `ibi` and `y` arrays statically mapped into true L1 Shared Memory. Threads cooperatively phase-fold the lightcurves, bin them atomically into shared memory, and search subsets of the permutations. A final tree-based reduction condenses the 128 answers down to the global arrays.
* **Result**: Lowered evaluation time to **`6.02s`**. Total speedup from baseline (7.93s) is now **24% Faster**.

## 4. Jupyter Notebook Benchmarking
* **Enhancement**: Added `%%time` commands inside `bls_cuda_v2.ipynb` to easily track Wall Time for performance evaluations.

## 5. Phase 3 Precision Limits (`bls_cuda_v2_f32.py`)
* **Experiment**: The GTX 1080 Ti operates FP32 math up to 32x faster than FP64 math. Created a dedicated clone `bls_cuda_v2_f32.py` enforcing strictly 32-bit float memory pipelines and math constraints across the board to measure precision decay on transit SNR.
* **Findings**: Signal-to-Noise Ratio (SNR) for both TESS and Kepler algorithms degraded by exactly `0.000` down to the third decimal place. The peak `power` scalar shifted by `< 0.1%`. Speedup varied heavily (1.44x faster on TESS data, only 1.04x faster on Kepler data) pointing to a primary Global-Memory-Bandwidth loading bottleneck over a true mathematics bottleneck.

---

### Date: 2026-11-25
### Target: `pytfit5/bls_gpu.py` (the GPU kernel moves into the package)

Merged `main` (C kernel, `dips_only`, variable bin counts `nbs`, the current normalisation) and wrote
the kernel again inside the package, using the same per-frequency logic as the C kernel:

1. **Same bins as the CPU.** Phases in FP64 with the CPU's operations.
2. **Exact, order-independent binning** (the GPU version of the C kernel's paired (flux, count)
   binning): one int64 per point, fixed-point flux in the high bits and a count of 1 in the low bits,
   added with one shared-memory integer atomic. Results are deterministic.
3. **FP32 search with the C kernel's division-free comparison.** The GPU tested (RTX PRO 4000 Blackwell
   SFF, 70 W) runs FP64 at 1/31 of the FP32 rate.
4. **Variable nb.** Frequencies are launched in groups by nb, with shared memory sized to the group;
   nb above ~12,000 (long periods with 5-min bins over two seasons) uses global memory.
5. **Numba code generation.** The inner loops use unsigned indices and while loops; signed `range`
   loops made Numba add a wraparound check and 64-bit iterator bookkeeping to every box, which
   doubled the search time. The wrap-around at nb is a split of the loop instead of a branch, and
   each bin's (flux, count) is one complex64 load.
6. **Tried and dropped:** double-float (FP32 pair) phases with an FP64 recheck near bin edges were
   slower than FP64 phases; binning is limited by reading the points, not by FP64.

Results (simulated GBTDS, production grid, 1.0 Msun star; C kernel = AVX-512 on Skylake-SP):

| | season 1 (21,348 freqs) | seasons 1+2 (47,416 freqs) |
|---|---|---|
| GPU kernel | 0.019 s | 0.074 s |
| C kernel, one core | 4.2 s | 20 s |
| same (jn1, jn2) as C | 100% | 100% |
| max p relative difference | 7e-8 | 9e-8 |

Production pipeline (roman_bench.py: two windows x 5 passes + pulse search), 48 representative stars,
light curves per hour: season 1: 7,600 / 10,200 / 10,100 with 4 / 8 / 12 CPU workers sharing the GPU
(C kernel, 8 workers: 500); seasons 1+2: 2,600 / 2,900 / 2,800. The GPU saturates at about 8 workers;
it then runs at its 70 W cap (~1.5-1.7 GHz).
