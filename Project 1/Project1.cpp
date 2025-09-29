#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <cstdlib>

// aligned allocation
template <typename T>
T* alloc_aligned(size_t N, size_t align=32) {
    void* ptr;
    posix_memalign(&ptr, align, N * sizeof(T));
    return (T*)ptr;
}

// ---------------- Kernels ----------------
template <typename T>
void saxpy_scalar(T a, const std::vector<T>& x, std::vector<T>& y, size_t stride=1) {
    for (size_t i = 0; i < x.size(); i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

template <typename T>
void saxpy_simd(T a, const std::vector<T>& x, std::vector<T>& y, size_t stride=1) {
    // Same loop — but compiled with -O3 -march=native will auto-vectorize
    for (size_t i = 0; i < x.size(); i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

template <typename T>
T dot_scalar(const std::vector<T>& x, const std::vector<T>& y, size_t stride=1) {
    T sum = 0.0;
    for (size_t i = 0; i < x.size(); i += stride) {
        sum += x[i] * y[i];
    }
    return sum;
}

template <typename T>
T dot_simd(const std::vector<T>& x, const std::vector<T>& y, size_t stride=1) {
    T sum = 0.0;
    for (size_t i = 0; i < x.size(); i += stride) {
        sum += x[i] * y[i];
    }
    return sum;
}

template <typename T>
void multiply_scalar(const std::vector<T>& x, const std::vector<T>& y, std::vector<T>& z, size_t stride=1) {
    for (size_t i = 0; i < x.size(); i += stride) {
        z[i] = x[i] * y[i];
    }
}

template <typename T>
void multiply_simd(const std::vector<T>& x, const std::vector<T>& y, std::vector<T>& z, size_t stride=1) {
    for (size_t i = 0; i < x.size(); i += stride) {
        z[i] = x[i] * y[i];
    }
}

// ---------------- Timer with stdev ----------------
template <typename Func>
std::pair<double,double> time_function(Func f, int repetitions = 5) {
    std::vector<double> times;
    for (int r = 0; r < repetitions; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(end-start).count());
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / repetitions;
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum/repetitions - mean*mean);
    return {mean, stdev};
}

// ---------------- Correctness check ----------------
template <typename T>
bool validate(const std::vector<T>& a, const std::vector<T>& b, double tol=1e-6) {
    for (size_t i=0; i<a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// ---------------- Main ----------------
int main() {
    using T = float; // change to double for datatype comparison
    std::vector<size_t> sizes = {1024, 32768, 1048576, 16777216};
    std::vector<size_t> strides = {1,2,4,8}; // stride sweep
    T a = 2.5;

    double cpu_freq_GHz = 3.0;
    double cpu_freq_Hz = cpu_freq_GHz * 1e9;

    std::mt19937 rng(42);
    std::uniform_real_distribution<T> dist(0.0, 1.0);

    std::ofstream out("results.csv");
    out << "N,Stride,Kernel,Scalar_GFLOPs,SIMD_GFLOPs,Speedup,"
        << "Scalar_CPE,SIMD_CPE,Scalar_Stdev,SIMD_Stdev\n";

    // ---------------- Normal experiments loop ----------------
    for (size_t N : sizes) {
        for (size_t stride : strides) {
            std::vector<T> x(N), y(N), z(N);
            for (size_t i = 0; i < N; i++) {
                x[i] = dist(rng);
                y[i] = dist(rng);
                z[i] = 0.0;
            }

            // ---------- SAXPY ----------
            auto y_scalar = y, y_simd = y;
            auto [ts_mean, ts_std] = time_function([&]() { saxpy_scalar(a, x, y_scalar, stride); });
            auto [tv_mean, tv_std] = time_function([&]() { saxpy_simd(a, x, y_simd, stride); });

            double flops = 2.0 * (N/stride);
            double gflops_scalar = (flops / ts_mean) / 1e9;
            double gflops_simd   = (flops / tv_mean) / 1e9;

            out << N << "," << stride << ",SAXPY," << gflops_scalar << "," << gflops_simd << ","
                << (ts_mean / tv_mean) << ","
                << (ts_mean * cpu_freq_Hz / (N/stride)) << ","
                << (tv_mean * cpu_freq_Hz / (N/stride)) << ","
                << ts_std << "," << tv_std << "\n";

            // ---------- DOT ----------
            auto [tds_mean, tds_std] = time_function([&]() { volatile T s = dot_scalar(x, y, stride); });
            auto [tdv_mean, tdv_std] = time_function([&]() { volatile T s = dot_simd(x, y, stride); });

            flops = 2.0 * (N/stride);
            gflops_scalar = (flops / tds_mean) / 1e9;
            gflops_simd   = (flops / tdv_mean) / 1e9;

            out << N << "," << stride << ",Dot," << gflops_scalar << "," << gflops_simd << ","
                << (tds_mean / tdv_mean) << ","
                << (tds_mean * cpu_freq_Hz / (N/stride)) << ","
                << (tdv_mean * cpu_freq_Hz / (N/stride)) << ","
                << tds_std << "," << tdv_std << "\n";

            // ---------- Multiply ----------
            auto z_scalar = z, z_simd = z;
            auto [tms_mean, tms_std] = time_function([&]() { multiply_scalar(x, y, z_scalar, stride); });
            auto [tmv_mean, tmv_std] = time_function([&]() { multiply_simd(x, y, z_simd, stride); });

            flops = 1.0 * (N/stride);
            gflops_scalar = (flops / tms_mean) / 1e9;
            gflops_simd   = (flops / tmv_mean) / 1e9;

            out << N << "," << stride << ",Multiply," << gflops_scalar << "," << gflops_simd << ","
                << (tms_mean / tmv_mean) << ","
                << (tms_mean * cpu_freq_Hz / (N/stride)) << ","
                << (tmv_mean * cpu_freq_Hz / (N/stride)) << ","
                << tms_std << "," << tmv_std << "\n";
        }
    }

    out.close();

    // ---------------- Alignment & Tail Test ----------------
    {
        size_t N1 = 1024;   // perfectly divisible by vector width
        size_t N2 = 1023;   // has a "tail" leftover

        // Allocate aligned arrays
        float* x = alloc_aligned<float>(N1);
        float* y = alloc_aligned<float>(N1);
        float* y_misaligned = (float*)((char*)y + 4); // shift for misalignment

        for (size_t i = 0; i < N1; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        auto [t_aligned, s_aligned] = time_function([&]() {
            for (size_t i = 0; i < N1; i++) {
                y[i] = a * x[i] + y[i];
            }
        });
        double gflops_aligned = (2.0 * N1 / t_aligned) / 1e9;

        auto [t_misaligned, s_misaligned] = time_function([&]() {
            for (size_t i = 0; i < N1; i++) {
                y_misaligned[i] = a * x[i] + y_misaligned[i];
            }
        });
        double gflops_misaligned = (2.0 * N1 / t_misaligned) / 1e9;

        auto [t_tail, s_tail] = time_function([&]() {
            for (size_t i = 0; i < N2; i++) {
                y[i] = a * x[i] + y[i];
            }
        });
        double gflops_tail = (2.0 * N2 / t_tail) / 1e9;

        std::ofstream out_align("alignment_tail.csv");
        out_align << "Case,GFLOPs\n";
        out_align << "Aligned," << gflops_aligned << "\n";
        out_align << "Misaligned," << gflops_misaligned << "\n";
        out_align << "Tail(N=1023)," << gflops_tail << "\n";
        out_align.close();

        std::cout << "Alignment/tail results written to alignment_tail.csv\n";
    }

    std::cout << "Results written to results.csv\n";
    return 0;
}
