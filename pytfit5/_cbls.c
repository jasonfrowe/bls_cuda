/*
 * Fast CPU BLS kernel for pytfit5 (C, OpenMP, AVX2 / AVX-512 intrinsics).
 *
 * Loaded with ctypes from pytfit5/bls_c.py. Built by setup.py as an optional
 * extension; the Python module stub at the bottom only exists so the build
 * system can produce and locate the shared library.
 *
 * Results (p, jn1, jn2) are bit-identical to pytfit5.bls_cpu.bls_kernel:
 *   - binning uses the same phase formula and per-bin summation order
 *     (flux and count accumulated as one (flux, 1.0) pair per point);
 *   - for each start bin i, widths m = 0..kma are visited in order and the
 *     first strictly better power is kept; start bins are combined in
 *     increasing order with strict '>' (ties resolve like the reference).
 * The search is register-blocked: a block of start bins keeps its running sums
 * S, K and its best (N[,D]) and E in vector registers for all widths.
 *
 * Variants (search):
 *   0 scalar        portable C, division
 *   1 avx2_div      pow = s^2/dfac, pow > B
 *   2 avx2_nodiv    s^2 * Dbest > Nbest * dfac (no divide)
 *   3 avx512_div
 *   4 avx512_nodiv
 * The library is compiled for baseline x86-64; ISA-specific functions carry
 * target attributes and are only called when the CPU supports them.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define CBLS_X86 1
#include <immintrin.h>
#else
#define CBLS_X86 0
#endif

#define G_CONST 6.674e-11
#define RSUN (696265.0e0 * 1000.0e0)
#define MSUN 1.9891e30
#define PIFAC 1.083852140278e0
#define DAY2SEC 86400.0
#define ONEHOUR (1.0 / 24.0)
#define PAD 64 /* extra doubles past 2*nb so tail blocks may over-read */

/* Same transit-duration limits as bls_cpu.bls_kernel */
static void limits(double freq, int nb, int npt, double Rstar, double Mstar,
                   int *kmi_, int *kma_, int *kkmi_)
{
    double fsec = freq / DAY2SEC;
    double q = PIFAC * Rstar * RSUN / pow(G_CONST * Mstar * MSUN, 1.0 / 3.0) * pow(fsec, 2.0 / 3.0);
    double qmi = q / 2.0;
    if (qmi < ONEHOUR * freq / 2) qmi = ONEHOUR * freq / 2;
    double qma = q * 2.0;
    if (qma > 0.25) qma = 0.25;
    if (qmi > qma) {
        qma = qmi * 2.0;
        if (qma > 0.4) qma = 0.4;
    }
    int kmi = (int)(qmi * nb);
    if (kmi < 1) kmi = 1;
    int kma = (int)(qma * nb) + 1;
    int kkmi = (int)(npt * qmi);
    if (kkmi < 5) kkmi = 5;
    *kmi_ = kmi; *kma_ = kma; *kkmi_ = kkmi;
}

/* ------------------------------------------------------------- binning -- */
static inline void phase_bins(double freq, const double *time, int npt, int nb, int32_t *jbin)
{
    double fnbm1 = (double)(nb - 1);
    for (int i = 0; i < npt; i++) {
        double tf = time[i] * freq;
        double phase = tf - (double)(int32_t)tf;
        jbin[i] = (int32_t)(phase * fnbm1);
    }
}

/* pairs: 2*nb doubles; ibi, y: 2*nb + PAD doubles (duplicated for wrap-around) */
static inline void scatter_pairs(const double *flux, const int32_t *jbin, int npt, int nb,
                                 double *ibi, double *y, double *pairs)
{
    memset(pairs, 0, 2 * (size_t)nb * sizeof(double));
#if CBLS_X86
    const __m128d one = _mm_set_pd(1.0, 0.0);
    for (int i = 0; i < npt; i++) {
        double *q = pairs + 2 * (size_t)jbin[i];
        _mm_storeu_pd(q, _mm_add_pd(_mm_loadu_pd(q), _mm_add_pd(_mm_load_sd(flux + i), one)));
    }
#else
    for (int i = 0; i < npt; i++) {
        double *q = pairs + 2 * (size_t)jbin[i];
        q[0] += flux[i];
        q[1] += 1.0;
    }
#endif
    for (int j = 0; j < nb; j++) {
        y[j] = y[j + nb] = pairs[2 * j];
        ibi[j] = ibi[j + nb] = pairs[2 * j + 1];
    }
    memset(y + 2 * (size_t)nb, 0, PAD * sizeof(double));
    memset(ibi + 2 * (size_t)nb, 0, PAD * sizeof(double));
}

static void bin_generic(double freq, const double *time, const double *flux, int npt, int nb,
                        double *ibi, double *y, int32_t *jbin, double *pairs)
{
    phase_bins(freq, time, npt, nb, jbin);
    scatter_pairs(flux, jbin, npt, nb, ibi, y, pairs);
}

#if CBLS_X86
__attribute__((target("avx2,fma")))
static void bin_avx2(double freq, const double *time, const double *flux, int npt, int nb,
                     double *ibi, double *y, int32_t *jbin, double *pairs)
{
    phase_bins(freq, time, npt, nb, jbin);
    scatter_pairs(flux, jbin, npt, nb, ibi, y, pairs);
}

__attribute__((target("avx512f,avx512dq,avx512vl")))
static void bin_avx512(double freq, const double *time, const double *flux, int npt, int nb,
                       double *ibi, double *y, int32_t *jbin, double *pairs)
{
    phase_bins(freq, time, npt, nb, jbin);
    scatter_pairs(flux, jbin, npt, nb, ibi, y, pairs);
}
#endif

/* -------------------------------------------------------------- search -- */
static void search_scalar(const double *ibi, const double *y, int nb, int npt,
                          int kmi, int kma, int kkmi, double smax,
                          double *p_, int *jn1_, int *jn2_)
{
    double p = 0.0; int jn1 = 0, jn2 = 0;
    double fnpt = npt, fkkmi = kkmi;
    for (int i = 0; i < nb; i++) {
        double s = 0, kk = 0, B = 0; int E = 0;
        for (int m = 0; m <= kma; m++) {
            s += y[i + m]; kk += ibi[i + m];
            if (m + 1 > kmi) {
                double dfac = kk * (fnpt - kk);
                double pw = s * s / dfac;
                if (kk > fkkmi && dfac > 0.0 && s < smax && pw > B) { B = pw; E = i + m; }
            }
        }
        if (B > p) { p = B; jn1 = i; jn2 = E; }
    }
    *p_ = p; *jn1_ = jn1; *jn2_ = jn2;
}

#if CBLS_X86
#define NV2 3 /* ymm vectors per block (4 lanes each) */
__attribute__((target("avx2,fma")))
static inline void search_avx2_impl(const double *ibi, const double *y, int nb, int npt,
                                    int kmi, int kma, int kkmi, const int usediv, double smax,
                                    double *p_, int *jn1_, int *jn2_)
{
    const __m256d vnpt = _mm256_set1_pd((double)npt);
    const __m256d vsmax = _mm256_set1_pd(smax);
    const __m256d vkkmi = _mm256_set1_pd((double)kkmi);
    const __m256d zero = _mm256_setzero_pd();
    double p = 0.0; int jn1 = 0, jn2 = 0;
    const int W = 4 * NV2;
    for (int i0 = 0; i0 < nb; i0 += W) {
        __m256d S[NV2], K[NV2], N[NV2], D[NV2], E[NV2];
        for (int v = 0; v < NV2; v++) {
            S[v] = zero; K[v] = zero; N[v] = zero; E[v] = zero;
            D[v] = usediv ? zero : _mm256_set1_pd(1.0);
        }
        int m = 0;
        for (; m <= kma && m + 1 <= kmi; m++)
            for (int v = 0; v < NV2; v++) {
                S[v] = _mm256_add_pd(S[v], _mm256_loadu_pd(y + i0 + 4 * v + m));
                K[v] = _mm256_add_pd(K[v], _mm256_loadu_pd(ibi + i0 + 4 * v + m));
            }
        for (; m <= kma; m++) {
            const __m256d vm = _mm256_set1_pd((double)m);
            for (int v = 0; v < NV2; v++) {
                __m256d s = _mm256_add_pd(S[v], _mm256_loadu_pd(y + i0 + 4 * v + m));
                __m256d kk = _mm256_add_pd(K[v], _mm256_loadu_pd(ibi + i0 + 4 * v + m));
                S[v] = s; K[v] = kk;
                __m256d dfac = _mm256_mul_pd(kk, _mm256_sub_pd(vnpt, kk));
                __m256d s2 = _mm256_mul_pd(s, s);
                __m256d ok = _mm256_and_pd(_mm256_cmp_pd(kk, vkkmi, _CMP_GT_OQ),
                                           _mm256_cmp_pd(dfac, zero, _CMP_GT_OQ));
                ok = _mm256_and_pd(ok, _mm256_cmp_pd(s, vsmax, _CMP_LT_OQ));
                __m256d better;
                if (usediv) {
                    __m256d pw = _mm256_div_pd(s2, dfac);
                    better = _mm256_and_pd(ok, _mm256_cmp_pd(pw, N[v], _CMP_GT_OQ));
                    N[v] = _mm256_blendv_pd(N[v], pw, better);
                } else {
                    better = _mm256_and_pd(ok, _mm256_cmp_pd(_mm256_mul_pd(s2, D[v]),
                                                             _mm256_mul_pd(N[v], dfac), _CMP_GT_OQ));
                    N[v] = _mm256_blendv_pd(N[v], s2, better);
                    D[v] = _mm256_blendv_pd(D[v], dfac, better);
                }
                E[v] = _mm256_blendv_pd(E[v], vm, better);
            }
        }
        double nb_[4 * NV2], db_[4 * NV2], eb_[4 * NV2];
        for (int v = 0; v < NV2; v++) {
            _mm256_storeu_pd(nb_ + 4 * v, N[v]);
            _mm256_storeu_pd(db_ + 4 * v, D[v]);
            _mm256_storeu_pd(eb_ + 4 * v, E[v]);
        }
        for (int t = 0; t < W && i0 + t < nb; t++) {
            double b = usediv ? nb_[t] : nb_[t] / db_[t];
            if (b > p) { p = b; jn1 = i0 + t; jn2 = i0 + t + (int)eb_[t]; }
        }
    }
    *p_ = p; *jn1_ = jn1; *jn2_ = jn2;
}

#define NV5 4 /* zmm vectors per block (8 lanes each) */
__attribute__((target("avx512f,avx512dq")))
static inline void search_avx512_impl(const double *ibi, const double *y, int nb, int npt,
                                      int kmi, int kma, int kkmi, const int usediv, double smax,
                                      double *p_, int *jn1_, int *jn2_)
{
    const __m512d vnpt = _mm512_set1_pd((double)npt);
    const __m512d vsmax = _mm512_set1_pd(smax);
    const __m512d vkkmi = _mm512_set1_pd((double)kkmi);
    const __m512d zero = _mm512_setzero_pd();
    double p = 0.0; int jn1 = 0, jn2 = 0;
    const int W = 8 * NV5;
    for (int i0 = 0; i0 < nb; i0 += W) {
        __m512d S[NV5], K[NV5], N[NV5], D[NV5], E[NV5];
        for (int v = 0; v < NV5; v++) {
            S[v] = zero; K[v] = zero; N[v] = zero; E[v] = zero;
            D[v] = usediv ? zero : _mm512_set1_pd(1.0);
        }
        int m = 0;
        for (; m <= kma && m + 1 <= kmi; m++)
            for (int v = 0; v < NV5; v++) {
                S[v] = _mm512_add_pd(S[v], _mm512_loadu_pd(y + i0 + 8 * v + m));
                K[v] = _mm512_add_pd(K[v], _mm512_loadu_pd(ibi + i0 + 8 * v + m));
            }
        for (; m <= kma; m++) {
            const __m512d vm = _mm512_set1_pd((double)m);
            for (int v = 0; v < NV5; v++) {
                __m512d s = _mm512_add_pd(S[v], _mm512_loadu_pd(y + i0 + 8 * v + m));
                __m512d kk = _mm512_add_pd(K[v], _mm512_loadu_pd(ibi + i0 + 8 * v + m));
                S[v] = s; K[v] = kk;
                __m512d dfac = _mm512_mul_pd(kk, _mm512_sub_pd(vnpt, kk));
                __m512d s2 = _mm512_mul_pd(s, s);
                __mmask8 ok = _mm512_cmp_pd_mask(kk, vkkmi, _CMP_GT_OQ);
                ok = _mm512_mask_cmp_pd_mask(ok, dfac, zero, _CMP_GT_OQ);
                ok = _mm512_mask_cmp_pd_mask(ok, s, vsmax, _CMP_LT_OQ);
                __mmask8 better;
                if (usediv) {
                    __m512d pw = _mm512_div_pd(s2, dfac);
                    better = _mm512_mask_cmp_pd_mask(ok, pw, N[v], _CMP_GT_OQ);
                    N[v] = _mm512_mask_mov_pd(N[v], better, pw);
                } else {
                    better = _mm512_mask_cmp_pd_mask(ok, _mm512_mul_pd(s2, D[v]),
                                                     _mm512_mul_pd(N[v], dfac), _CMP_GT_OQ);
                    N[v] = _mm512_mask_mov_pd(N[v], better, s2);
                    D[v] = _mm512_mask_mov_pd(D[v], better, dfac);
                }
                E[v] = _mm512_mask_mov_pd(E[v], better, vm);
            }
        }
        double nb_[8 * NV5], db_[8 * NV5], eb_[8 * NV5];
        for (int v = 0; v < NV5; v++) {
            _mm512_storeu_pd(nb_ + 8 * v, N[v]);
            _mm512_storeu_pd(db_ + 8 * v, D[v]);
            _mm512_storeu_pd(eb_ + 8 * v, E[v]);
        }
        for (int t = 0; t < W && i0 + t < nb; t++) {
            double b = usediv ? nb_[t] : nb_[t] / db_[t];
            if (b > p) { p = b; jn1 = i0 + t; jn2 = i0 + t + (int)eb_[t]; }
        }
    }
    *p_ = p; *jn1_ = jn1; *jn2_ = jn2;
}

__attribute__((target("avx2,fma")))
static void search_avx2_div(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double smax, double *p, int *j1, int *j2)
{ search_avx2_impl(a, b, nb, npt, kmi, kma, kkmi, 1, smax, p, j1, j2); }
__attribute__((target("avx2,fma")))
static void search_avx2_nodiv(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double smax, double *p, int *j1, int *j2)
{ search_avx2_impl(a, b, nb, npt, kmi, kma, kkmi, 0, smax, p, j1, j2); }
__attribute__((target("avx512f,avx512dq")))
static void search_avx512_div(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double smax, double *p, int *j1, int *j2)
{ search_avx512_impl(a, b, nb, npt, kmi, kma, kkmi, 1, smax, p, j1, j2); }
__attribute__((target("avx512f,avx512dq")))
static void search_avx512_nodiv(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double smax, double *p, int *j1, int *j2)
{ search_avx512_impl(a, b, nb, npt, kmi, kma, kkmi, 0, smax, p, j1, j2); }
#endif /* CBLS_X86 */

typedef void (*search_fn)(const double *, const double *, int, int, int, int, int, double, double *, int *, int *);
typedef void (*bin_fn)(double, const double *, const double *, int, int, double *, double *, int32_t *, double *);

/* ------------------------------------------------------------ public API -- */

/* Bit mask of variants this CPU can run: bit v set => variant v usable. */
int cbls_supported_variants(void)
{
    int mask = 1; /* scalar */
#if CBLS_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        mask |= (1 << 1) | (1 << 2);
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl"))
        mask |= (1 << 3) | (1 << 4);
#endif
    return mask;
}

int cbls_has_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

/* Returns 0 on success, -1 if the variant is not supported on this CPU,
 * -2 on allocation failure.
 * A box counts only if its flux sum s < smax: smax = 0 keeps dips only (flux is
 * median-centred, so a transit has s < 0); smax = +INFINITY keeps both signs,
 * i.e. the classic s^2 statistic (bit-identical to cbls_scan). */
int cbls_scan_ex(const double *freqs, const int64_t *nbs, int nfreq, const double *time,
                 const double *flux, int npt, double Rstar, double Mstar, int variant,
                 int nthreads, double smax, double *p, int32_t *jn1, int32_t *jn2)
{
    if (variant < 0 || variant > 4 || !((cbls_supported_variants() >> variant) & 1))
        return -1;
    search_fn fn = search_scalar;
    bin_fn bfn = bin_generic;
#if CBLS_X86
    switch (variant) {
    case 1: fn = search_avx2_div; bfn = bin_avx2; break;
    case 2: fn = search_avx2_nodiv; bfn = bin_avx2; break;
    case 3: fn = search_avx512_div; bfn = bin_avx512; break;
    case 4: fn = search_avx512_nodiv; bfn = bin_avx512; break;
    }
#endif
    int nbmax = 1;
    for (int i = 0; i < nfreq; i++)
        if (nbs[i] > nbmax) nbmax = (int)nbs[i];
    int err = 0;
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = omp_get_max_threads();
#pragma omp parallel num_threads(nthreads) reduction(| : err)
#endif
    {
        double *ibi = malloc((2 * (size_t)nbmax + PAD) * sizeof(double));
        double *y = malloc((2 * (size_t)nbmax + PAD) * sizeof(double));
        double *pairs = malloc((2 * (size_t)nbmax) * sizeof(double));
        int32_t *jbin = malloc((size_t)npt * sizeof(int32_t));
        int ok = ibi && y && pairs && jbin;
        if (!ok) err = 1;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (int i = 0; i < nfreq; i++) {
            if (!ok) continue;
            int nb = (int)nbs[i];
            bfn(freqs[i], time, flux, npt, nb, ibi, y, jbin, pairs);
            int kmi, kma, kkmi, a, b;
            limits(freqs[i], nb, npt, Rstar, Mstar, &kmi, &kma, &kkmi);
            fn(ibi, y, nb, npt, kmi, kma, kkmi, smax, &p[i], &a, &b);
            jn1[i] = a;
            jn2[i] = b;
        }
        free(ibi); free(y); free(pairs); free(jbin);
    }
    return err ? -2 : 0;
}

/* Original entry point (both signs), kept for ABI compatibility. */
int cbls_scan(const double *freqs, const int64_t *nbs, int nfreq, const double *time,
              const double *flux, int npt, double Rstar, double Mstar, int variant,
              int nthreads, double *p, int32_t *jn1, int32_t *jn2)
{
    return cbls_scan_ex(freqs, nbs, nfreq, time, flux, npt, Rstar, Mstar, variant,
                        nthreads, INFINITY, p, jn1, jn2);
}

#ifdef CBLS_PYMODULE
/* Empty Python module so setuptools can build/locate this library. */
#include <Python.h>
static struct PyModuleDef cbls_module = {PyModuleDef_HEAD_INIT, "_cbls",
                                         "ctypes-loaded BLS kernels", -1, NULL};
PyMODINIT_FUNC PyInit__cbls(void) { return PyModule_Create(&cbls_module); }
#endif
