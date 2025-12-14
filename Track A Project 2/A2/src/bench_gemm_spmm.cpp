#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "matrix_gen.hpp"
#include "gemm.hpp"
#include "spmm.hpp"
#include "timing.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Max absolute difference between two dense matrices
double max_abs_diff(const std::vector<float>& A,
                    const std::vector<float>& B) {
    double maxd = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        maxd = std::max(maxd, std::abs(static_cast<double>(A[i] - B[i])));
    }
    return maxd;
}

// Convert CSR -> dense row-major (m x k)
std::vector<float> csr_to_dense(const CSR& A) {
    std::vector<float> dense(A.m * A.k, 0.0f);
    for (int i = 0; i < A.m; ++i) {
        for (int p = A.row_ptr[i]; p < A.row_ptr[i + 1]; ++p) {
            int j = A.col_idx[p];
            dense[i * A.k + j] = A.values[p];
        }
    }
    return dense;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " m k n density kernel threads reps\n";
        std::cerr << "  kernel = dense | dense_scalar | dense_simd | spmm | check\n";
        return 1;
    }

    int m       = std::stoi(argv[1]);
    int k       = std::stoi(argv[2]);
    int n       = std::stoi(argv[3]);
    double dens = std::stod(argv[4]);
    std::string kernel = argv[5];
    int threads = std::stoi(argv[6]);
    int reps    = std::stoi(argv[7]);

#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads; // avoid unused warning without OpenMP
#endif

    unsigned long long seed = 42;

    // ---------------------------
    // Correctness mode
    // ---------------------------
    if (kernel == "check") {
        // Use small sizes for sanity check
        if (m <= 0) m = 64;
        if (k <= 0) k = 64;
        if (n <= 0) n = 64;

        double test_dens = (dens > 0.0) ? dens : 0.1;

        // Generate a random sparse matrix A_csr and convert to dense
        CSR A_csr = generate_csr_random(m, k, test_dens, -1.0f, 1.0f, seed);
        std::vector<float> A_dense = csr_to_dense(A_csr);
        std::vector<float> B = make_dense_random(k, n, -1.0f, 1.0f, seed + 1);

        std::vector<float> C_naive(m * n);
        std::vector<float> C_block(m * n);
        std::vector<float> C_spmm(m * n);

        // Reference: naive dense GEMM on densified A
        gemm_naive(A_dense.data(), B.data(), C_naive.data(), m, k, n);

        // Blocked + OpenMP GEMM on same dense A
        gemm_blocked_omp(A_dense.data(), B.data(), C_block.data(),
                         m, k, n,
                         64, 64, 64);

        // SpMM on CSR A
#ifdef _OPENMP
        spmm_csr_omp(A_csr, B.data(), C_spmm.data(), n);
#else
        spmm_csr_single(A_csr, B.data(), C_spmm.data(), n);
#endif

        double err_block = max_abs_diff(C_naive, C_block);
        double err_spmm  = max_abs_diff(C_naive, C_spmm);

        std::cout << "Correctness check (m=" << m
                  << ", k=" << k << ", n=" << n << ", density=" << test_dens << ")\n";
        std::cout << "Max abs error (blocked_omp vs naive) = " << err_block << "\n";
        std::cout << "Max abs error (SpMM vs naive)        = " << err_spmm  << "\n";

        return 0;
    }

    // ---------------------------
    // Performance modes
    // ---------------------------

    // B is always dense k x n
    std::vector<float> B = make_dense_random(k, n, -1.0f, 1.0f, seed + 1);
    std::vector<float> C(m * n);

    double best_time = 1e100;
    double flops     = 0.0;
    std::size_t nnz  = 0;

    // Dense variants: scalar, SIMD, SIMD+threads
    if (kernel == "dense" ||
        kernel == "dense_scalar" ||
        kernel == "dense_simd") {

        std::vector<float> A = make_dense_random(m, k, -1.0f, 1.0f, seed);
        flops = 2.0 * static_cast<double>(m) * k * n;
        nnz   = static_cast<std::size_t>(m) * k;

        for (int r = 0; r < reps; ++r) {
            double t0 = now_seconds();

            if (kernel == "dense_scalar") {
                // pure scalar triple loop
                gemm_naive(A.data(), B.data(), C.data(), m, k, n);
            } else if (kernel == "dense_simd") {
                // blocked, single-thread (SIMD from -O3)
                gemm_blocked(A.data(), B.data(), C.data(),
                             m, k, n,
                             64, 64, 64);
            } else { // "dense" = blocked + OpenMP
                gemm_blocked_omp(A.data(), B.data(), C.data(),
                                 m, k, n,
                                 64, 64, 64);
            }

            double t1 = now_seconds();
            double dt = t1 - t0;
            if (dt < best_time) best_time = dt;
        }

    } else if (kernel == "spmm") {
        // Sparse A in CSR: m x k, with density dens
        CSR A = generate_csr_random(m, k, dens, -1.0f, 1.0f, seed);
        nnz = A.values.size();
        flops = 2.0 * static_cast<double>(nnz) * n;

        for (int r = 0; r < reps; ++r) {
            double t0 = now_seconds();
#ifdef _OPENMP
            spmm_csr_omp(A, B.data(), C.data(), n);
#else
            spmm_csr_single(A, B.data(), C.data(), n);
#endif
            double t1 = now_seconds();
            double dt = t1 - t0;
            if (dt < best_time) best_time = dt;
        }

    } else {
        std::cerr << "Unknown kernel: " << kernel << "\n";
        return 1;
    }

    double gflops = flops / best_time / 1e9;

    // CSV: m,k,n,density,kernel,threads,best_time_s,GFLOP_s,nnz
    std::cout << m << "," << k << "," << n << ","
              << dens << ","
              << kernel << ","
              << threads << ","
              << best_time << ","
              << gflops << ","
              << nnz << "\n";

    return 0;
}
