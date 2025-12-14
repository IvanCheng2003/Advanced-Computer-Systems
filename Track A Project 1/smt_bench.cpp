// smt_bench.cpp
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
#include <chrono>

using namespace std;

double now_s() {
    using clock = chrono::steady_clock;
    return chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void compute_kernel(float *a, size_t n, long iters, int tid, int nthreads) {
    size_t chunk = (n + nthreads - 1) / nthreads;
    size_t start = tid * chunk;
    size_t end   = min(n, start + chunk);
    if (start >= end) return;

    float s0 = 1.0f + tid;
    float s1 = 2.0f + tid;

    for (long t = 0; t < iters; ++t) {
        for (size_t i = start; i < end; ++i) {
            s0 = s0 * 1.000001f + a[i];
            s1 = s1 * 0.999999f + a[i];
        }
    }
    // prevent dead-code elimination
    a[start] = s0 + s1;
}

void memory_kernel(float *a, size_t n, long iters, int tid, int nthreads) {
    size_t chunk = (n + nthreads - 1) / nthreads;
    size_t start = tid * chunk;
    size_t end   = min(n, start + chunk);
    if (start >= end) return;

    for (long t = 0; t < iters; ++t) {
        for (size_t i = start; i < end; ++i) {
            a[i] = a[i] * 1.000001f + 1.0f;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " compute|memory num_threads array_size iters\n";
        return 1;
    }

    string mode = argv[1];
    int nthreads = stoi(argv[2]);
    size_t n = stoull(argv[3]);
    long iters = stol(argv[4]);

    cout << "Mode=" << mode
         << " threads=" << nthreads
         << " n=" << n
         << " iters=" << iters << "\n";

    // allocate array (page aligned)
    float *a = nullptr;
    if (posix_memalign((void**)&a, 4096, n * sizeof(float)) != 0) {
        cerr << "posix_memalign failed\n";
        return 1;
    }
    for (size_t i = 0; i < n; ++i) a[i] = 1.0f;

    auto kernel = (mode == "compute") ? compute_kernel : memory_kernel;

    double t0 = now_s();

    vector<thread> ths;
    for (int t = 0; t < nthreads; ++t) {
        ths.emplace_back(kernel, a, n, iters, t, nthreads);
    }
    for (auto &th : ths) th.join();

    double t1 = now_s();
    double dt = t1 - t0;

    double total_ops = (double)n * (double)iters;
    double ops_per_s = total_ops / dt;

    cout << "Time_s=" << dt << "\n";
    cout << "Total_updates=" << total_ops << "\n";
    cout << "Updates_per_s=" << ops_per_s << "\n";

    free(a);
    return 0;
}
