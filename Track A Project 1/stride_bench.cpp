// stride_bench.cpp
#include <bits/stdc++.h>
#include <chrono>

using namespace std;

double now_s() {
    using clock = chrono::steady_clock;
    return chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " megabytes stride_bytes iters warmup_iters\n";
        cerr << "Example: " << argv[0] << " 256 64 20 2\n";
        return 1;
    }

    double megabytes = atof(argv[1]);
    size_t stride_bytes = stoull(argv[2]);
    long iters = stol(argv[3]);
    long warmup_iters = stol(argv[4]);

    size_t N_bytes = (size_t)(megabytes * (1ull << 20));
    size_t Nelems = N_bytes / sizeof(float);
    size_t stride_elems = max<size_t>(1, stride_bytes / sizeof(float));

    cout << "Allocating " << megabytes << " MiB "
         << "(bytes=" << N_bytes << ", elements=" << Nelems << ")\n";
    cout << "Stride_bytes=" << stride_bytes
         << " (stride_elems=" << stride_elems << ") "
         << "iters=" << iters
         << " warmup_iters=" << warmup_iters << "\n";

    float *a = nullptr;
    if (posix_memalign((void**)&a, 4096, Nelems * sizeof(float)) != 0) {
        cerr << "posix_memalign failed\n";
        return 1;
    }

    for (size_t i = 0; i < Nelems; ++i) a[i] = 1.0f;

    volatile float sink = 0.0f;

    auto run_once = [&](long reps) {
        for (long it = 0; it < reps; ++it) {
            for (size_t i = 0; i < Nelems; i += stride_elems) {
                sink += a[i];
            }
        }
    };

    // warmup
    run_once(warmup_iters);

    double t0 = now_s();
    run_once(iters);
    double t1 = now_s();

    double dt = t1 - t0;

    size_t touches_per_iter = (Nelems + stride_elems - 1) / stride_elems;
    double total_touches = (double)touches_per_iter * iters;
    double bytes_touched = total_touches * sizeof(float);

    cout << "Time_s=" << dt << "\n";
    cout << "Touches=" << total_touches << "\n";
    cout << "Bytes_touched=" << bytes_touched << "\n";
    cout << "Bandwidth_GBps=" << (bytes_touched / dt) / 1e9 << "\n";
    cout << "sink=" << sink << "\n";

    free(a);
    return 0;
}
