#include "gemm.hpp"
#include <algorithm>
#include <cstring>
#include <omp.h>

// Naive GEMM: C = A * B
void gemm_naive(const float* A, const float* B, float* C,
                int m, int k, int n) {
    std::memset(C, 0, sizeof(float) * m * n);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            float a = A[i * k + p];
            const float* b_row = &B[p * n];
            float* c_row = &C[i * n];
            for (int j = 0; j < n; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

// Blocked GEMM, single-threaded
void gemm_blocked(const float* A, const float* B, float* C,
                  int m, int k, int n,
                  int Bm, int Bk, int Bn) {
    std::memset(C, 0, sizeof(float) * m * n);
    for (int i0 = 0; i0 < m; i0 += Bm) {
        int i_max = std::min(i0 + Bm, m);
        for (int p0 = 0; p0 < k; p0 += Bk) {
            int p_max = std::min(p0 + Bk, k);
            for (int j0 = 0; j0 < n; j0 += Bn) {
                int j_max = std::min(j0 + Bn, n);
                for (int i = i0; i < i_max; ++i) {
                    float* c_row = &C[i * n];
                    for (int p = p0; p < p_max; ++p) {
                        float a = A[i * k + p];
                        const float* b_row = &B[p * n];
                        for (int j = j0; j < j_max; ++j) {
                            c_row[j] += a * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

// Blocked GEMM with OpenMP
void gemm_blocked_omp(const float* A, const float* B, float* C,
                      int m, int k, int n,
                      int Bm, int Bk, int Bn) {
    std::memset(C, 0, sizeof(float) * m * n);

    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < m; i0 += Bm) {
        int i_max = std::min(i0 + Bm, m);
        for (int p0 = 0; p0 < k; p0 += Bk) {
            int p_max = std::min(p0 + Bk, k);
            for (int j0 = 0; j0 < n; j0 += Bn) {
                int j_max = std::min(j0 + Bn, n);
                for (int i = i0; i < i_max; ++i) {
                    float* c_row = &C[i * n];
                    for (int p = p0; p < p_max; ++p) {
                        float a = A[i * k + p];
                        const float* b_row = &B[p * n];
                        for (int j = j0; j < j_max; ++j) {
                            c_row[j] += a * b_row[j];
                        }
                    }
                }
            }
        }
    }
}
