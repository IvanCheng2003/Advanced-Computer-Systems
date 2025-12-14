#pragma once

// Naive triple-loop GEMM: C = A * B
// A: m x k, B: k x n, C: m x n, row-major
void gemm_naive(const float* A, const float* B, float* C,
                int m, int k, int n);

// Blocked GEMM, single-threaded (SIMD-friendly via -O3)
// Block sizes Bm, Bk, Bn
void gemm_blocked(const float* A, const float* B, float* C,
                  int m, int k, int n,
                  int Bm, int Bk, int Bn);

// Blocked GEMM with OpenMP over outer i-blocks
void gemm_blocked_omp(const float* A, const float* B, float* C,
                      int m, int k, int n,
                      int Bm, int Bk, int Bn);
