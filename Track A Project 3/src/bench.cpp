#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "hashtable.hpp"

using clk = std::chrono::steady_clock;

struct Args {
  std::string impl = "coarse";      // coarse|striped
  std::string workload = "lookup";  // lookup|insert|mix7030
  uint64_t nkeys = 10000;
  int threads = 1;
  double seconds = 2.0;
  uint64_t seed = 123;
  uint64_t nlocks = 1024;
  int prefill = 1;
  int run_id = 1;
};

static void die(const std::string &msg) {
  std::cerr << "Error: " << msg << "\n";
  std::exit(1);
}

static bool get_flag(int &i, int argc, char **argv, const char *name, std::string &out) {
  if (std::strcmp(argv[i], name) == 0) {
    if (i + 1 >= argc) die(std::string("Missing value for ") + name);
    out = argv[++i];
    return true;
  }
  return false;
}

static bool get_flag(int &i, int argc, char **argv, const char *name, uint64_t &out) {
  if (std::strcmp(argv[i], name) == 0) {
    if (i + 1 >= argc) die(std::string("Missing value for ") + name);
    out = std::stoull(argv[++i]);
    return true;
  }
  return false;
}

static bool get_flag(int &i, int argc, char **argv, const char *name, int &out) {
  if (std::strcmp(argv[i], name) == 0) {
    if (i + 1 >= argc) die(std::string("Missing value for ") + name);
    out = std::stoi(argv[++i]);
    return true;
  }
  return false;
}

static bool get_flag(int &i, int argc, char **argv, const char *name, double &out) {
  if (std::strcmp(argv[i], name) == 0) {
    if (i + 1 >= argc) die(std::string("Missing value for ") + name);
    out = std::stod(argv[++i]);
    return true;
  }
  return false;
}

static Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s;
    if (get_flag(i, argc, argv, "--impl", s)) { a.impl = s; continue; }
    if (get_flag(i, argc, argv, "--workload", s)) { a.workload = s; continue; }
    if (get_flag(i, argc, argv, "--nkeys", a.nkeys)) continue;
    if (get_flag(i, argc, argv, "--threads", a.threads)) continue;
    if (get_flag(i, argc, argv, "--seconds", a.seconds)) continue;
    if (get_flag(i, argc, argv, "--seed", a.seed)) continue;
    if (get_flag(i, argc, argv, "--nlocks", a.nlocks)) continue;
    if (get_flag(i, argc, argv, "--prefill", a.prefill)) continue;
    if (get_flag(i, argc, argv, "--run_id", a.run_id)) continue;

    if (std::strcmp(argv[i], "--help") == 0) {
      std::cout <<
        "Usage: ./a4_bench --impl coarse|striped --workload lookup|insert|mix7030\n"
        "                --nkeys N --threads T --seconds S --seed X --nlocks L --prefill 0|1 --run_id R\n";
      std::exit(0);
    }

    die(std::string("Unknown arg: ") + argv[i]);
  }

  if (a.impl != "coarse" && a.impl != "striped") die("--impl must be coarse|striped");
  if (a.workload != "lookup" && a.workload != "insert" && a.workload != "mix7030")
    die("--workload must be lookup|insert|mix7030");
  if (a.threads <= 0) die("--threads must be > 0");
  if (a.seconds <= 0) die("--seconds must be > 0");
  if (a.nkeys == 0) die("--nkeys must be > 0");
  if (a.nlocks == 0) a.nlocks = 1;
  return a;
}

static std::vector<uint64_t> make_keys(uint64_t n, uint64_t seed, uint64_t salt) {
  std::vector<uint64_t> keys(n);
  for (uint64_t i = 0; i < n; ++i) {
    keys[i] = mix64(seed ^ (salt + i * 0x9e3779b97f4a7c15ULL));
  }
  return keys;
}

int main(int argc, char **argv) {
  Args a = parse_args(argc, argv);

  size_t nbuckets = static_cast<size_t>(a.nkeys * 2 + 1);

  std::unique_ptr<IHashTable> ht;
  if (a.impl == "coarse") {
    ht.reset(new HashTableCoarse(nbuckets));
  } else {
    ht.reset(new HashTableStriped(nbuckets, static_cast<size_t>(a.nlocks)));
  }

  // Prefill keys and an "insert stream"
  auto prefill_keys = make_keys(a.nkeys, a.seed, 0xA1);
  auto insert_keys  = make_keys(a.nkeys * 4, a.seed, 0xB2); // larger stream for inserts

  if (a.prefill) {
    for (uint64_t i = 0; i < a.nkeys; ++i) {
      (void)ht->insert(prefill_keys[i], prefill_keys[i] ^ 0x12345678ULL);
    }
  }

  // Warm up briefly
  {
    uint64_t tmp = 0;
    auto t0 = clk::now();
    while (std::chrono::duration<double>(clk::now() - t0).count() < 0.2) {
      (void)ht->find(prefill_keys[0], tmp);
    }
  }

  std::atomic<bool> start_flag{false};
  std::atomic<bool> stop_flag{false};

  std::vector<uint64_t> thread_ops(a.threads, 0);

  auto worker = [&](int tid) {
    std::mt19937_64 rng(mix64(a.seed ^ (uint64_t)tid));
    std::uniform_int_distribution<int> pct(0, 99);
    std::uniform_int_distribution<uint64_t> pick_pref(0, a.nkeys - 1);
    std::uniform_int_distribution<uint64_t> pick_ins(0, insert_keys.size() - 1);

    uint64_t local_ops = 0;
    uint64_t tmp = 0;

    // wait for start
    while (!start_flag.load(std::memory_order_acquire)) { /* spin */ }

    while (!stop_flag.load(std::memory_order_relaxed)) {
      if (a.workload == "lookup") {
        uint64_t k = prefill_keys[pick_pref(rng)];
        (void)ht->find(k, tmp);
      } else if (a.workload == "insert") {
        // Make inserts mostly unique per thread by mixing tid into key stream index
        uint64_t idx = (uint64_t)local_ops * (uint64_t)a.threads + (uint64_t)tid;
        uint64_t k = insert_keys[idx % insert_keys.size()];
        (void)ht->insert(k, k ^ 0xCAFEBABEULL);
      } else { // mix7030: 70 find, 15 insert, 15 erase
        int r = pct(rng);
        if (r < 70) {
          uint64_t k = prefill_keys[pick_pref(rng)];
          (void)ht->find(k, tmp);
        } else if (r < 85) {
          uint64_t idx = (uint64_t)local_ops * (uint64_t)a.threads + (uint64_t)tid;
          uint64_t k = insert_keys[idx % insert_keys.size()];
          (void)ht->insert(k, k ^ 0xDEADBEEFULL);
        } else {
          uint64_t k = prefill_keys[pick_pref(rng)];
          (void)ht->erase(k);
        }
      }

      ++local_ops;
    }

    thread_ops[tid] = local_ops;
  };

  // Launch threads
  std::vector<std::thread> th;
  th.reserve(a.threads);
  for (int t = 0; t < a.threads; ++t) th.emplace_back(worker, t);

  // Start timing window
  auto t_start = clk::now();
  start_flag.store(true, std::memory_order_release);

  while (std::chrono::duration<double>(clk::now() - t_start).count() < a.seconds) {
    // busy wait to avoid sleep jitter
  }
  stop_flag.store(true, std::memory_order_release);

  for (auto &t : th) t.join();

  uint64_t total_ops = 0;
  for (int t = 0; t < a.threads; ++t) total_ops += thread_ops[t];

  double elapsed = std::chrono::duration<double>(clk::now() - t_start).count();
  double ops_per_s = (elapsed > 0) ? (double)total_ops / elapsed : 0.0;

  std::cout
    << a.impl << ","
    << a.workload << ","
    << a.nkeys << ","
    << a.threads << ","
    << a.nlocks << ","
    << a.seconds << ","
    << a.seed << ","
    << a.run_id << ","
    << total_ops << ","
    << ops_per_s
    << "\n";

  return 0;
}
