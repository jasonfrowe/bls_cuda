"""Count search cells (start bin x width) per frequency for the bench grid."""
import sys, numpy as np
sys.path.insert(0, "avx512")
from bench import load_lc, freq_grid, DEFAULT_LC
from kernels import _limits
t, f = load_lc(DEFAULT_LC)
freqs, nb = freq_grid(t)
freqs = freqs[1.0 / freqs < t.max()][::50]
kma = np.array([_limits(fr, nb, t.size, 1.0, 1.0)[1] for fr in freqs])
cells = nb * (kma + 1)
print(f"freqs={freqs.size} mean kma={kma.mean():.1f} median kma={np.median(kma):.0f} max={kma.max()}")
print(f"mean cells/freq={cells.mean():.3e}  points/freq={t.size}")
