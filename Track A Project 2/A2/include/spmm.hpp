#pragma once
#include "csr.hpp"

void spmm_csr_single(const CSR& A,
                     const float* B, float* C,
                     int n);

void spmm_csr_omp(const CSR& A,
                  const float* B, float* C,
                  int n);
