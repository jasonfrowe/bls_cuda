# AVX-512 BLS investigation

The production kernel is `pytfit5/_cbls.c`, loaded by `pytfit5/bls_c.py` and used by
`bls_cpu.bls()` when `bls_engine` is `"auto"` or `"c"`. This directory holds the
prototypes and benchmarks that led to it.

## Findings on kona (2x Xeon Silver 4116, Skylake-SP, 24C/48T)

- Numba only emits 256-bit (ymm) code on Skylake-AVX512, because LLVM's tuning prefers
  256 bits. Forcing 512-bit with `NUMBA_CPU_FEATURES="<host features>,-prefer-256-bit"`
  is 20-40% **slower**, because AVX-512 lowers the clock and Silver has one 512-bit FMA port.
- What helps:
  1. **Register-blocked search.** Each block of start bins keeps S, K, best and end bin in
     registers for all widths.
  2. **Division-free compare** for AVX-512: `s^2*D_best > N_best*dfac`. `vdivpd` has the
     same per-element throughput at both widths, so the division gains nothing from zmm.
  3. **Interleaved (flux, count) pair binning**, one 128-bit add per point. This stays
     bit-identical.
- Full `bls()` on KOI-1 (51,832 points, 771,774 frequencies, 48 threads):
  Numba 23.3 s, C `avx512_nodiv` 12.2 s (1.91x), C `avx2_div` 15.9 s (1.47x).
  Four more KOIs gave 1.89-1.95x. All results are identical to Numba.
- With division, AVX-512 is slower than AVX2. Without the binning fix, the AVX-512 gain
  disappears at 48 threads (hyperthreading).

## Testing on another machine (e.g. DRAC rorqual)

Rorqual's CPUs implement AVX-512 differently from kona's Xeon Silver (no comparable clock
penalty, different divide and port layout), so the best variant may differ. Compare all of them.

```bash
# on a login node
module load python/3.12            # plus a gcc module if needed
virtualenv --no-download ~/venv-bls && source ~/venv-bls/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy numba scipy matplotlib tqdm pandas h5py pytest  # astroquery may need PyPI
git clone -b avx512-bls https://github.com/jasonfrowe/bls_cuda.git && cd bls_cuda
pip install -e . --no-deps          # builds pytfit5/_cbls (check that it compiled:)
python -c "from pytfit5 import bls_c; print(bls_c.supported_variants(), bls_c.resolve_variant())"

# copy a Kepler LC over, e.g. from kona: /data/Kepler/Kepler_n/koi1.n/klc11446443.dc.dat

# in an allocation (thread count follows the Slurm CPU allocation)
salloc --cpus-per-task=64 --mem=16G --time=1:00:00
PYTFIT5_TEST_LC=klc11446443.dc.dat python -m pytest tests/test_bls_c.py -q
python avx512/bls_e2e.py klc11446443.dc.dat avx512_nodiv avx512_div avx2_div avx2_nodiv scalar

# single-core Numba 256 vs 512-bit codegen
python avx512/bench.py klc11446443.dc.dat
F=$(python -c "import llvmlite.binding as b; print(b.get_host_cpu_features().flatten())")
NUMBA_CPU_FEATURES="$F,-prefer-256-bit" NUMBA_CACHE_DIR=/tmp/nbc512 python avx512/bench.py klc11446443.dc.dat
```

If a variant other than `avx512_nodiv` wins there, set `bls_isa` explicitly, or change
`AUTO_ORDER` in `pytfit5/bls_c.py`.

## Files

- `cbls.c`, `cbls.py`, `build.sh`: prototype kernels, including binning-only and search-only modes.
- `kernels.py`: Numba variants (tiled, division-free).
- `bench.py`: single-core per-kernel timing. `scan_bench.py`: OpenMP prototype scan.
  `prod_bench.py`: compares against the Numba ProcessPool. `bls_e2e.py`: full `bls()` with each engine.
- `inspect_asm.py`: counts xmm/ymm/zmm instructions in Numba's generated code.
