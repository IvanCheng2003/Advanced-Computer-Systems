// affinity_bench.cpp
#define _GNU_SOURCE
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <sched.h>

using namespace std;

double now_s() {
    using clock = chrono::steady_clock;
    return chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void pin_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(),
                                    sizeof(cpu_set_t),
                                    &cpuset);
    if (rc != 0) {
        cerr << "Warning: pthread_setaffinity_np failed for CPU "
             << cpu_id << " (errno=" << rc << ")\n";
    }
}

// simple CPU-bound work: add/multiply loop
void worker(long long iters, int cpu_id, atomic<long long> &checksum) {
    pin_thread_to_cpu(cpu_id);

    long long local = 0;
    for (long long i = 0; i < iters; ++i) {
        local += (i * 13LL) ^ (i >> 3);
    }
    checksum.fetch_add(local, memory_order_relaxed);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " num_threads cpu0 cpu1 ... [iters_per_thread]\n";
        cerr << "Example: " << argv[0] << " 4 0 1 2 3 100000000\n";
        return 1;
    }

    int nthreads = stoi(argv[1]);
    if (argc < 2 + nthreads) {
        cerr << "Error: need " << nthreads << " CPU IDs\n";
        return 1;
    }

    long long iters = 100000000LL;
    if (argc >= 2 + nthreads + 1) {
        iters = stoll(argv[2 + nthreads]);
    }

    vector<int> cpus(nthreads);
    for (int i = 0; i < nthreads; ++i) {
        cpus[i] = stoi(argv[2 + i]);
    }

    cout << "Threads=" << nthreads << " iters_per_thread=" << iters << "\n";
    cout << "CPU mapping: ";
    for (int i = 0; i < nthreads; ++i) cout << cpus[i] << (i + 1 == nthreads ? '\n' : ' ');

    atomic<long long> checksum{0};

    double t0 = now_s();

    vector<thread> ths;
    for (int i = 0; i < nthreads; ++i) {
        ths.emplace_back(worker, iters, cpus[i], ref(checksum));
    }
    for (auto &th : ths) th.join();

    double t1 = now_s();
    double dt = t1 - t0;

    long long total_work = iters * 1LL * nthreads;
    double iters_per_s = (double)total_work / dt;

    cout << "Time_s=" << dt << "\n";
    cout << "Total_iters=" << total_work << "\n";
    cout << "Iters_per_s=" << iters_per_s << "\n";
    cout << "Checksum=" << checksum.load() << "\n";

    return 0;
}
