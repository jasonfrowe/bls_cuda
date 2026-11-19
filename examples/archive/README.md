# Archived notebooks

These notebooks were written for earlier versions of the code and are kept for reference only. They
are not maintained and may not run against the current API (for example, they predate the C BLS
kernel, the `dips_only` and `max_bin_minutes` options, the fixed period validation and the compiled
detrender).

| Notebook | What it was |
|---|---|
| `pytfit5_example.ipynb` | The original pytfit5 walk-through (detrending, BLS, fitting, MCMC) |
| `pytfit5_BOI-1.ipynb` | The same workflow applied to one candidate (BOI-1) |
| `backup_code.ipynb` | Development scratch code for period validation |
| `bls_cuda.ipynb`, `bls_cuda_v2.ipynb` | The CUDA (GPU) BLS scripts in `cudaFiles/` |

For current usage, see [`../roman_example.ipynb`](../roman_example.ipynb) and [API.md](../../API.md).
