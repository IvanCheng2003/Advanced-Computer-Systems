#include "spmm.hpp"
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

void spmm_csr_single(const CSR& A,
                     const float* B, float* C,
                     int n) {
    std::memset(C, 0, sizeof(float) * A.m * n);
    for (int i = 0; i < A.m; ++i) {
        float* c_row = &C[i * n];
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int k = A.col_idx[p];
            float a = A.values[p];
            const float* b_row = &B[k * n];
            for (int j = 0; j < n; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

void spmm_csr_omp(const CSR& A,
                  const float* B, float* C,
                  int n) {
    std::memset(C, 0, sizeof(float) * A.m * n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.m; ++i) {
        float* c_row = &C[i * n];
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int k = A.col_idx[p];
            float a = A.values[p];
            const float* b_row = &B[k * n];
            for (int j = 0; j < n; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}
