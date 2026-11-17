/*
 * Experimental BLS kernels in C with explicit AVX2 / AVX-512 intrinsics.
 *
 * The algorithm and arithmetic match pytfit5.bls_cpu.bls_kernel:
 *   - binning identical (same phase formula, same summation order)
 *   - for each start bin i, widths m = 0..kma are accumulated in order and the
 *     first strictly-better power is kept; start bins are combined in increasing
 *     order with strict '>' (so ties resolve like bls_kernel_reference).
 * The search is register-blocked: a block of start bins keeps its running sums
 * S, K and best (B | N,D), E in vector registers for all widths m, so the hot
 * loop only does two unaligned loads (y, ibi) per vector per width.
 *
 * usediv=1: pow = s^2/dfac, compare pow > B          (bit-identical to bls_kernel)
 * usediv=0: compare s^2 * Dbest > Nbest * dfac        (division-free)
 *
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC cbls.c -o libcbls.so -lm
 */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define G_CONST 6.674e-11
#define RSUN (696265.0e0 * 1000.0e0)
#define MSUN 1.9891e30
#define PIFAC 1.083852140278e0
#define DAY2SEC 86400.0
#define ONEHOUR (1.0 / 24.0)
#define PAD 64 /* extra doubles past 2*nb so tail blocks may over-read */

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

/* ibi, y: length >= 2*nb + PAD, zeroed by caller */
static void bin_data(double freq, const double *time, const double *flux, int npt,
                     int nb, double *ibi, double *y, int32_t *jbin)
{
    double fnbm1 = (double)(nb - 1);
    for (int i = 0; i < npt; i++) {
        double tf = time[i] * freq;
        double phase = tf - (double)(int32_t)tf;
        jbin[i] = (int32_t)(phase * fnbm1);
    }
    for (int i = 0; i < npt; i++) {
        int j = jbin[i];
        ibi[j] += 1.0;
        y[j] += flux[i];
    }
    memcpy(ibi + nb, ibi, nb * sizeof(double));
    memcpy(y + nb, y, nb * sizeof(double));
}

/* Interleaved binning: accumulate (flux, 1.0) pairs with one 128-bit add per
 * point (same per-bin summation order, so bit-identical), then de-interleave. */
static inline void bin_scatter_pairs(const double *flux, const int32_t *jbin, int npt,
                                     int nb, double *ibi, double *y, double *pairs)
{
    memset(pairs, 0, 2 * (size_t)nb * sizeof(double));
    const __m128d one = _mm_set_pd(1.0, 0.0);
    for (int i = 0; i < npt; i++) {
        double *q = pairs + 2 * (size_t)jbin[i];
        __m128d v = _mm_add_pd(_mm_loadu_pd(q), _mm_add_pd(_mm_load_sd(flux + i), one));
        _mm_storeu_pd(q, v);
    }
    for (int j = 0; j < nb; j++) {
        y[j] = y[j + nb] = pairs[2 * j];
        ibi[j] = ibi[j + nb] = pairs[2 * j + 1];
    }
}

__attribute__((target("avx2,fma")))
static void bin_pairs_avx2(double freq, const double *time, const double *flux, int npt,
                           int nb, double *ibi, double *y, int32_t *jbin, double *pairs)
{
    double fnbm1 = (double)(nb - 1);
    for (int i = 0; i < npt; i++) {
        double tf = time[i] * freq;
        double phase = tf - (double)(int32_t)tf;
        jbin[i] = (int32_t)(phase * fnbm1);
    }
    bin_scatter_pairs(flux, jbin, npt, nb, ibi, y, pairs);
}

__attribute__((target("avx512f,avx512dq,avx512vl")))
static void bin_pairs_avx512(double freq, const double *time, const double *flux, int npt,
                             int nb, double *ibi, double *y, int32_t *jbin, double *pairs)
{
    double fnbm1 = (double)(nb - 1);
    for (int i = 0; i < npt; i++) {
        double tf = time[i] * freq;
        double phase = tf - (double)(int32_t)tf;
        jbin[i] = (int32_t)(phase * fnbm1);
    }
    bin_scatter_pairs(flux, jbin, npt, nb, ibi, y, pairs);
}

/* ---------------------------------------------------------------- scalar -- */
static void search_scalar(const double *ibi, const double *y, int nb, int npt,
                          int kmi, int kma, int kkmi, double *p_, int *jn1_, int *jn2_)
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
                if (kk > fkkmi && dfac > 0.0 && pw > B) { B = pw; E = i + m; }
            }
        }
        if (B > p) { p = B; jn1 = i; jn2 = E; }
    }
    *p_ = p; *jn1_ = jn1; *jn2_ = jn2;
}

/* ------------------------------------------------------------------ AVX2 -- */
#define NV2 3 /* ymm vectors per block (4 lanes each) */
__attribute__((target("avx2,fma")))
static inline void search_avx2_impl(const double *ibi, const double *y, int nb, int npt,
                                    int kmi, int kma, int kkmi, const int usediv,
                                    double *p_, int *jn1_, int *jn2_)
{
    const __m256d vnpt = _mm256_set1_pd((double)npt);
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

/* --------------------------------------------------------------- AVX-512 -- */
#define NV5 4 /* zmm vectors per block (8 lanes each) */
__attribute__((target("avx512f,avx512dq")))
static inline void search_avx512_impl(const double *ibi, const double *y, int nb, int npt,
                                      int kmi, int kma, int kkmi, const int usediv,
                                      double *p_, int *jn1_, int *jn2_)
{
    const __m512d vnpt = _mm512_set1_pd((double)npt);
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
static void search_avx2_div(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double *p, int *j1, int *j2)
{ search_avx2_impl(a, b, nb, npt, kmi, kma, kkmi, 1, p, j1, j2); }
__attribute__((target("avx2,fma")))
static void search_avx2_nodiv(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double *p, int *j1, int *j2)
{ search_avx2_impl(a, b, nb, npt, kmi, kma, kkmi, 0, p, j1, j2); }
__attribute__((target("avx512f,avx512dq")))
static void search_avx512_div(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double *p, int *j1, int *j2)
{ search_avx512_impl(a, b, nb, npt, kmi, kma, kkmi, 1, p, j1, j2); }
__attribute__((target("avx512f,avx512dq")))
static void search_avx512_nodiv(const double *a, const double *b, int nb, int npt, int kmi, int kma, int kkmi, double *p, int *j1, int *j2)
{ search_avx512_impl(a, b, nb, npt, kmi, kma, kkmi, 0, p, j1, j2); }

typedef void (*search_fn)(const double *, const double *, int, int, int, int, int, double *, int *, int *);
static search_fn pick(int variant)
{
    switch (variant) {
    case 1: return search_avx2_div;
    case 2: return search_avx2_nodiv;
    case 3: return search_avx512_div;
    case 4: return search_avx512_nodiv;
    default: return search_scalar;
    }
}

/* variant: 0 scalar, 1 avx2 div, 2 avx2 nodiv, 3 avx512 div, 4 avx512 nodiv
 * mode:    0 full kernel, 1 binning only, 2 search only (on dummy bins) */
void bls_kernel_c(double freq, const double *time, const double *flux, int npt, int nb,
                  double Rstar, double Mstar, int variant, int mode, double *out)
{
    double *ibi = calloc(2 * (size_t)nb + PAD, sizeof(double));
    double *y = calloc(2 * (size_t)nb + PAD, sizeof(double));
    int32_t *jbin = malloc((size_t)npt * sizeof(int32_t));
    double p = 0; int jn1 = 0, jn2 = 0;
    if (mode != 2) {
        bin_data(freq, time, flux, npt, nb, ibi, y, jbin);
    } else {
        for (int i = 0; i < 2 * nb; i++) { ibi[i] = (double)npt / nb; y[i] = sin(i * 0.01) * 1e-4; }
    }
    if (mode != 1) {
        int kmi, kma, kkmi;
        limits(freq, nb, npt, Rstar, Mstar, &kmi, &kma, &kkmi);
        pick(variant)(ibi, y, nb, npt, kmi, kma, kkmi, &p, &jn1, &jn2);
    } else {
        p = y[0];
    }
    out[0] = p; out[1] = jn1; out[2] = jn2;
    free(ibi); free(y); free(jbin);
}

/* Full scan over nfreq frequencies (Period < baseline path), OpenMP-parallel. */
void bls_scan_c(const double *freqs, const int64_t *nbs, int nfreq, const double *time,
                const double *flux, int npt, double Rstar, double Mstar, int variant,
                double *p, int32_t *jn1, int32_t *jn2)
{
    int binmode = variant >> 4;  /* 0 original, 1 pairs */
    search_fn fn = pick(variant & 15);
#pragma omp parallel
    {
        int nbmax = 0;
        for (int i = 0; i < nfreq; i++) if (nbs[i] > nbmax) nbmax = (int)nbs[i];
        double *ibi = malloc((2 * (size_t)nbmax + PAD) * sizeof(double));
        double *y = malloc((2 * (size_t)nbmax + PAD) * sizeof(double));
        int32_t *jbin = malloc((size_t)npt * sizeof(int32_t));
        double *pairs = malloc((2 * (size_t)nbmax + PAD) * sizeof(double));
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < nfreq; i++) {
            int nb = (int)nbs[i];
            if (binmode == 0) {
                memset(ibi, 0, (2 * (size_t)nb + PAD) * sizeof(double));
                memset(y, 0, (2 * (size_t)nb + PAD) * sizeof(double));
                bin_data(freqs[i], time, flux, npt, nb, ibi, y, jbin);
            } else if ((variant & 15) >= 3) {
                bin_pairs_avx512(freqs[i], time, flux, npt, nb, ibi, y, jbin, pairs);
            } else {
                bin_pairs_avx2(freqs[i], time, flux, npt, nb, ibi, y, jbin, pairs);
            }
            int kmi, kma, kkmi;
            limits(freqs[i], nb, npt, Rstar, Mstar, &kmi, &kma, &kkmi);
            int a, b;
            fn(ibi, y, nb, npt, kmi, kma, kkmi, &p[i], &a, &b);
            jn1[i] = a; jn2[i] = b;
        }
        free(ibi); free(y); free(jbin); free(pairs);
    }
}
