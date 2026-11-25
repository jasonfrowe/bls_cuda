# cudaFiles

The GPU BLS kernel is now part of the package: `pytfit5/bls_gpu.py`, used by `bls()` with
`bls_engine="cuda"` (install with `pip install ".[cuda]"`; see API.md).

| File | What it is |
|---|---|
| `bench_gpu.py` | GPU vs C kernel on a simulated Roman GBTDS light curve: times, box agreement, power difference |
| `gpurun.sh` | Runs a command with GPU temperature, power and clock logging (one line per second) and kills it if the GPU reaches `LIMIT` °C (default 83) |
| `bls_cuda.py`, `bls_cuda_v1.py`, `bls_cuda_v2.py`, `bls_cuda_v2_f32.py`, `transitfind5_gpu.py`, `bls_cuda_v2.ipynb` | The original standalone CUDA BLS scripts (before 2026/11/25), kept for reference. They predate the current search grid, `nbs`, `dips_only` and the spectrum normalisation, so their results differ from `bls()` |
