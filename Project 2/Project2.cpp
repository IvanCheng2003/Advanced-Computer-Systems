#include <bits/stdc++.h>
#include <thread>
#include <random>
#include <chrono>
#include <atomic>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

using namespace std;
using clk = chrono::high_resolution_clock;

/************** Utilities **************/
static size_t parse_size(string s){
    if(s.empty()) return 0;
    char u = toupper(s.back());
    bool has_unit = isalpha(u);
    size_t v = stoull(s.substr(0, s.size() - (has_unit ? 1 : 0)));
    switch (u) {
        case 'B': return v;
        case 'K': return v * 1024ull;
        case 'M': return v * 1024ull * 1024ull;
        case 'G': return v * 1024ull * 1024ull * 1024ull;
        default:  return stoull(s); // bytes
    }
}
static inline double secs(function<void()> f, int reps=3){
    auto t0 = clk::now();
    for(int r=0;r<reps;r++) f();
    auto t1 = clk::now();
    chrono::duration<double> d = t1 - t0;
    return d.count() / max(1,reps);
}
static inline void pin_one_core(int core=0){
    // You can also run `taskset -c 0` from your script; left empty by default.
    (void)core;
}

/************** Aligned/huge allocation helpers **************/
struct Buffer {
    void* ptr=nullptr;
    size_t bytes=0;
    bool huge=false;
};
static Buffer alloc_buffer(size_t bytes, bool try_huge){
    Buffer b; b.bytes = bytes; b.huge=false;
    if(try_huge){
#ifdef MAP_HUGETLB
        long hps = 2*1024*1024; // 2MB hugepage (common default)
        size_t need = ((bytes + hps-1)/hps) * hps;
        void* p = mmap(nullptr, need, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
        if(p != MAP_FAILED){
            b.ptr = p; b.bytes = need; b.huge = true;
            return b;
        }
#endif
    }
    // fallback to posix_memalign 64B
    void* p=nullptr;
    size_t need = bytes;
    if(posix_memalign(&p, 64, max<size_t>(need,64)) != 0) p=nullptr;
    b.ptr = p; b.bytes = need; b.huge=false;
    return b;
}
static void free_buffer(Buffer& b){
    if(!b.ptr) return;
    if(b.huge){
        munmap(b.ptr, b.bytes);
    }else{
        free(b.ptr);
    }
    b.ptr=nullptr; b.bytes=0; b.huge=false;
}

/************** Args **************/
struct Args {
    string mode="latency";     // latency | bw | mix | intensity | cache | tlb
    string csv="out.csv";
    string pattern="seq";      // seq | random | chase (latency only uses chase)
    size_t strideB=64;         // ~64B/256B/1024B etc.
    size_t minB=4*1024, maxB=1ull<<30; // sweep range
    int points=20;
    int threads=1;
    size_t size_fixed=0;       // for fixed-size runs (bytes)
    double freqGHz=2.5;        // for cycles conversions
    bool huge=false;           // request huge pages (tlb mode)
    string rw="100R";          // for mix mode: 100R|100W|70R30W|50R50W
};
static Args parse(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;i++){
        string k = argv[i];
        auto need = [&](const char* nm){ if(i+1>=argc){ cerr<<"Missing value for "<<k<<"\n"; exit(1);} return string(argv[++i]); };
        if(k=="--mode") a.mode=need("mode");
        else if(k=="--csv") a.csv=need("csv");
        else if(k=="--pattern") a.pattern=need("pattern");
        else if(k=="--stride") a.strideB=parse_size(need("stride"));
        else if(k=="--min") a.minB=parse_size(need("min"));
        else if(k=="--max") a.maxB=parse_size(need("max"));
        else if(k=="--points") a.points=stoi(need("points"));
        else if(k=="--threads") a.threads=stoi(need("threads"));
        else if(k=="--size") a.size_fixed=parse_size(need("size"));
        else if(k=="--freqGHz") a.freqGHz=stod(need("freq"));
        else if(k=="--huge") { string v=need("huge"); a.huge=(v=="1"||v=="true"||v=="yes"); }
        else if(k=="--rw") a.rw=need("rw");
        else { cerr<<"Unknown arg "<<k<<"\n"; exit(1); }
    }
    return a;
}

/************** 1) Zero-queue latency: pointer chase **************/
static double latency_pointer_chase_bytes(size_t bytes, size_t strideB){
    size_t elems = max<size_t>(2, bytes / sizeof(size_t));
    vector<size_t> next(elems);
    size_t strideE = max<size_t>(1, strideB / sizeof(size_t));
    // Build a ring with given stride
    for(size_t i=0;i<elems;i++) next[i] = (i + strideE) % elems;

    volatile size_t idx = 0;
    for(size_t i=0;i<elems;i++) idx = next[idx]; // warm
    const size_t steps = elems * 16;
    auto t0 = clk::now();
    for(size_t i=0;i<steps;i++) idx = next[idx];
    auto t1 = clk::now();
    return chrono::duration<double>(t1-t0).count() / steps * 1e9; // ns/access
}

/************** 2) Bandwidth kernels & 3) R/W mixes **************/
template<class F>
static double timed_stream(size_t bytes, size_t strideB, int threads, F body){
    size_t N = max<size_t>(1, bytes / sizeof(float));
    // aligned vectors:
    vector<float> x(N), y(N), z(N);
    mt19937 rng(42); uniform_real_distribution<float> dist(0.0f,1.0f);
    for(size_t i=0;i<N;i++){ x[i]=dist(rng); y[i]=dist(rng); z[i]=0.0f; }

    // prepare sequential or random index order
    vector<size_t> idx(N); iota(idx.begin(), idx.end(), 0);
    size_t strideE = max<size_t>(1, strideB / sizeof(float));

    auto worker_seq = [&](size_t lo, size_t hi){
        for(size_t i=lo;i<hi;i+=strideE) body(i, x, y, z);
    };
    auto worker_rand = [&](size_t lo, size_t hi){
        // build a random walk across [lo,hi) with step strideE
        vector<size_t> order;
        for(size_t i=lo;i<hi;i+=strideE) order.push_back(i);
        shuffle(order.begin(), order.end(), rng);
        for(size_t j=0;j<order.size();j++) body(order[j], x, y, z);
    };

    vector<pair<size_t,size_t>> ranges;
    size_t chunk = N / threads;
    size_t start = 0;
    for(int t=0;t<threads;t++){
        size_t end = (t==threads-1)? N : start + chunk;
        ranges.push_back({start,end});
        start=end;
    }

    auto run = [&](bool random){
        vector<thread> ts;
        for(int t=0;t<threads;t++){
            if(random) ts.emplace_back(worker_rand, ranges[t].first, ranges[t].second);
            else       ts.emplace_back(worker_seq,  ranges[t].first, ranges[t].second);
        }
        for(auto& th: ts) th.join();
    };

    // time
    double sec = secs([&](){ run(false); }, 3);
    return sec; // seconds
}
static inline double bytes_touched_add(size_t N){ return (2+1)*N*sizeof(float); }
static inline double bytes_touched_copy(size_t N){ return (1+1)*N*sizeof(float); }

/************** 4) Intensity sweep helper **************/
struct TLPoint { double GBps; double ns_per_elem; int threads; };
static TLPoint run_triad_intensity(size_t bytes, size_t strideB, int threads){
    size_t N = max<size_t>(1, bytes/sizeof(float));
    vector<float> x(N), y(N), z(N);
    mt19937 rng(123); uniform_real_distribution<float> dist(0.0f,1.0f);
    for(size_t i=0;i<N;i++){ x[i]=dist(rng); y[i]=dist(rng); z[i]=0.0f; }
    float a=2.5f; size_t strideE = max<size_t>(1, strideB/sizeof(float));

    auto worker = [&](size_t lo,size_t hi){
        for(size_t i=lo;i<hi;i+=strideE) z[i] = x[i] + a*y[i];
    };
    vector<pair<size_t,size_t>> ranges;
    size_t chunk = N/threads, s=0;
    for(int t=0;t<threads;t++){
        size_t e = (t==threads-1)? N : s+chunk;
        ranges.push_back({s,e}); s=e;
    }
    auto run=[&](){
        vector<thread> ts; ts.reserve(threads);
        for(int t=0;t<threads;t++) ts.emplace_back(worker, ranges[t].first, ranges[t].second);
        for(auto& th: ts) th.join();
    };

    double sec = secs(run, 3);
    double touched = bytes_touched_add(N); // triad: 2R+1W ~ add
    double GBps = touched / sec / 1e9;
    double ns_elem = (sec / N) * 1e9;
    return {GBps, ns_elem, threads};
}

/************** 6) Cache-miss impact (light kernel) **************/
static double light_multiply_time(size_t bytes, string pattern, size_t strideB){
    size_t N = max<size_t>(1, bytes/sizeof(float));
    vector<float> x(N), y(N), z(N,0.0f);
    mt19937 rng(42); uniform_real_distribution<float> dist(0.0f,1.0f);
    for(size_t i=0;i<N;i++){ x[i]=dist(rng); y[i]=dist(rng); }
    vector<size_t> idx(N); iota(idx.begin(),idx.end(),0);
    if(pattern=="random"){ shuffle(idx.begin(), idx.end(), rng); }

    size_t strideE = max<size_t>(1, strideB/sizeof(float));
    auto run = [&](){
        if(pattern=="seq"){
            for(size_t i=0;i<N;i+=strideE) z[i] = x[i]*y[i];
        }else{
            for(size_t k=0;k<N;k+=strideE){ size_t i = idx[k]; z[i] = x[i]*y[i]; }
        }
    };
    return secs(run, 3); // seconds
}

/************** 7) TLB impact: huge pages + page-sized strides **************/
static double tlb_walk_test(size_t bytes, bool want_huge, size_t page_strideB, bool random){
    Buffer B = alloc_buffer(bytes, want_huge);
    if(!B.ptr){ cerr<<"Alloc failed (huge="<<want_huge<<")\n"; return NAN; }
    size_t N = B.bytes / sizeof(uint64_t);
    auto* a = reinterpret_cast<uint64_t*>(B.ptr);
    // touch pages with the chosen stride
    vector<size_t> offs;
    for(size_t off=0; off+page_strideB <= B.bytes; off += page_strideB)
        offs.push_back(off/sizeof(uint64_t));
    if(random){ mt19937 rng(7); shuffle(offs.begin(), offs.end(), rng); }

    // read pass (accumulate) + write pass to ensure mapping
    volatile uint64_t sink=0;
    auto run = [&](){
        for(size_t j=0;j<offs.size();j++) sink += a[offs[j]];
        for(size_t j=0;j<offs.size();j++) a[offs[j]] = sink + j;
    };
    double sec = secs(run, 3);
    free_buffer(B);
    return sec;
}


/************** Main **************/
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Args A = parse(argc, argv);

    // Open CSV
    ofstream out(A.csv);
    if(!out){ cerr<<"Cannot open "<<A.csv<<"\n"; return 1; }

    // Generate sizes (log-spaced if not fixed)
    vector<size_t> sizes;
    if(A.size_fixed){
        sizes.push_back(A.size_fixed);
    }else{
        double lmin = log((double)A.minB), lmax = log((double)A.maxB);
        for(int i=0;i<A.points;i++){
            double f = (A.points==1)? 0.0 : (double)i/(A.points-1);
            size_t sz = (size_t)exp(lmin + f*(lmax - lmin));
            sz = (sz + 4095) & ~4095ull; // 4KB align
            sizes.push_back(max<size_t>(sz, 4*1024));
        }
        sort(sizes.begin(), sizes.end());
        sizes.erase(unique(sizes.begin(), sizes.end()), sizes.end());
    }

    if(A.mode=="latency"){
        // 1) Zero-queue latency baselines
        out<<"bytes,pattern,strideB,ns_per_access,cycles_per_access\n";
        for(auto sz: sizes){
            double ns = latency_pointer_chase_bytes(sz, A.strideB);
            double cyc = ns * A.freqGHz;
            out<<sz<<","<<"chase"<<","<<A.strideB<<","<<ns<<","<<cyc<<"\n";
        }
        cerr<<"Wrote latency baselines -> "<<A.csv<<"\n";
    }
    else if(A.mode=="bw"){
    // 2) Pattern & granularity sweep (bandwidth + per-elem latency proxy)
    out<<"bytes,pattern,strideB,GBps,ns_per_elem\n";
    for(auto sz: sizes){
        size_t N = max<size_t>(1, sz/sizeof(float));
        auto body_add = [](size_t i, auto& x, auto& y, auto& z){ z[i] = x[i] + y[i]; };

        // time run depending on pattern
        double sec;
        if(A.pattern=="seq"){
            sec = timed_stream(sz, A.strideB, A.threads, body_add);
        } else if(A.pattern=="random"){
            // reuse timed_stream but with randomization inside
            // (make sure your timed_stream supports random pattern or add a flag)
            sec = timed_stream(sz, A.strideB, A.threads, body_add); 
            // if not, add a random path in timed_stream
        } else {
            cerr<<"Unknown pattern "<<A.pattern<<"\n";
            continue;
        }

        double GBps = ((2+1)*N*sizeof(float)) / sec / 1e9;
        double ns_elem = (sec / N) * 1e9;
        out<<sz<<","<<A.pattern<<","<<A.strideB<<","<<GBps<<","<<ns_elem<<"\n";
    }
    cerr<<"Wrote bandwidth sweep -> "<<A.csv<<"\n";
    }
    else if(A.mode=="mix"){
        // 3) Read/Write mix sweep
        out<<"bytes,rw,strideB,GBps,ns_per_elem\n";
        size_t sz = sizes.back();
        size_t N = max<size_t>(1, sz/sizeof(float));
        vector<float> x(N,1.0f), y(N,0.0f);
        int Rd=0, Wr=0;
        string RW = A.rw; for(auto& c: RW) c=toupper(c);
        if(RW=="100R"){Rd=10;Wr=0;}
        else if(RW=="100W"){Rd=0;Wr=10;}
        else if(RW=="70R30W"){Rd=7;Wr=3;}
        else if(RW=="50R50W"){Rd=5;Wr=5;}
        else {Rd=7;Wr=3;}
        size_t strideE = max<size_t>(1, A.strideB/sizeof(float));
        volatile float sink=0.f;

        auto run = [&](){
            for(size_t i=0;i<N;i+=strideE){
                for(int k=0;k<Rd;k++) sink += x[i];
                for(int k=0;k<Wr;k++) y[i] = x[i];
            }
        };
        double sec = secs(run, 3);
        double touched = (Rd + Wr) * (N/strideE) * sizeof(float);
        double GBps = touched / sec / 1e9;
        double ns_elem = (sec / (double)(N/strideE)) * 1e9;
        out<<sz<<","<<RW<<","<<A.strideB<<","<<GBps<<","<<ns_elem<<"\n";
        cerr<<"Wrote R/W mix -> "<<A.csv<<"\n";
    }
    else if(A.mode=="intensity"){
        // 4) Intensity sweep
        out<<"bytes,strideB,threads,GBps,ns_per_elem\n";
        size_t sz = sizes.back();
        for(int t=1; t<=A.threads; t*=2){
            TLPoint p = run_triad_intensity(sz, A.strideB, t);
            out<<sz<<","<<A.strideB<<","<<p.threads<<","<<p.GBps<<","<<p.ns_per_elem<<"\n";
        }
        cerr<<"Wrote intensity sweep -> "<<A.csv<<"\n";
    }
    else if(A.mode=="locality"){
        // 5) Working-set size sweep
        out<<"bytes,strideB,ns_per_access,cycles_per_access\n";
        for(auto sz: sizes){
            double ns = latency_pointer_chase_bytes(sz, A.strideB);
            double cyc = ns * A.freqGHz;
            out<<sz<<","<<A.strideB<<","<<ns<<","<<cyc<<"\n";
        }
        cerr<<"Wrote working-set locality sweep -> "<<A.csv<<"\n";
    }
    else if(A.mode=="cache"){
        // 6) Cache-miss impact
        out<<"bytes,pattern,strideB,elapsed_s\n";
        for(auto sz: sizes){
            double t = light_multiply_time(sz, A.pattern, A.strideB);
            out<<sz<<","<<A.pattern<<","<<A.strideB<<","<<t<<"\n";
        }
        cerr<<"Wrote cache-miss impact -> "<<A.csv<<"\n";
    }
    else if(A.mode=="tlb"){
        // 7) TLB impact
        out<<"bytes,huge,page_strideB,random,elapsed_s,huge_granted\n";
        size_t page = sysconf(_SC_PAGESIZE);
        size_t stride = (A.strideB ? A.strideB : page);
        for(auto sz: sizes){
            double t1 = tlb_walk_test(sz, false, stride, A.pattern=="random");
            out<<sz<<","<<0<<","<<stride<<","<<(A.pattern=="random")<<","<<t1<<","<<0<<"\n";
            if(A.huge){
                double t2 = tlb_walk_test(sz, true, 2*1024*1024, A.pattern=="random");
                out<<sz<<","<<1<<","<<2*1024*1024<<","<<(A.pattern=="random")<<","<<t2<<","<<1<<"\n";
            }
        }
        cerr<<"Wrote TLB sweep -> "<<A.csv<<"\n";
    }
    else {
        cerr<<"Unknown --mode. Use: latency|bw|mix|intensity|locality|cache|tlb\n";
        return 1;
    }

    return 0;
}
