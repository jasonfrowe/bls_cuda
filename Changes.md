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
