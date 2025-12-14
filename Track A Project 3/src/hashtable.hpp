#pragma once
#include <cstdint>
#include <vector>
#include <utility>
#include <mutex>

static inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct IHashTable {
  virtual ~IHashTable() = default;
  virtual bool insert(uint64_t key, uint64_t value) = 0;  // true if inserted new
  virtual bool find(uint64_t key, uint64_t &out_value) = 0;
  virtual bool erase(uint64_t key) = 0;                  // true if erased
};

class HashTableCoarse final : public IHashTable {
 public:
  explicit HashTableCoarse(size_t nbuckets)
      : buckets_(nbuckets) {}

  bool insert(uint64_t key, uint64_t value) override {
    std::lock_guard<std::mutex> g(mu_);
    auto &b = bucket_for_(key);
    for (auto &kv : b) {
      if (kv.first == key) { kv.second = value; return false; }
    }
    b.emplace_back(key, value);
    return true;
  }

  bool find(uint64_t key, uint64_t &out_value) override {
    std::lock_guard<std::mutex> g(mu_);
    auto &b = bucket_for_(key);
    for (auto &kv : b) {
      if (kv.first == key) { out_value = kv.second; return true; }
    }
    return false;
  }

  bool erase(uint64_t key) override {
    std::lock_guard<std::mutex> g(mu_);
    auto &b = bucket_for_(key);
    for (size_t i = 0; i < b.size(); ++i) {
      if (b[i].first == key) {
        b[i] = b.back();
        b.pop_back();
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> buckets_;
  std::mutex mu_;

  inline std::vector<std::pair<uint64_t, uint64_t>> &bucket_for_(uint64_t key) {
    uint64_t h = mix64(key);
    return buckets_[static_cast<size_t>(h % buckets_.size())];
  }
};


class HashTableStriped final : public IHashTable {
 public:
  HashTableStriped(size_t nbuckets, size_t nlocks)
      : buckets_(nbuckets),
        locks_(nlocks ? nlocks : 1) {}

  bool insert(uint64_t key, uint64_t value) override {
    const size_t bidx = bucket_index_(key);
    std::lock_guard<std::mutex> g(lock_for_bucket_(bidx));
    auto &b = buckets_[bidx];

    for (auto &kv : b) {
      if (kv.first == key) { kv.second = value; return false; }
    }
    b.emplace_back(key, value);
    return true;
  }

  bool find(uint64_t key, uint64_t &out_value) override {
    const size_t bidx = bucket_index_(key);
    std::lock_guard<std::mutex> g(lock_for_bucket_(bidx));
    auto &b = buckets_[bidx];

    for (auto &kv : b) {
      if (kv.first == key) { out_value = kv.second; return true; }
    }
    return false;
  }

  bool erase(uint64_t key) override {
    const size_t bidx = bucket_index_(key);
    std::lock_guard<std::mutex> g(lock_for_bucket_(bidx));
    auto &b = buckets_[bidx];

    for (size_t i = 0; i < b.size(); ++i) {
      if (b[i].first == key) {
        b[i] = b.back();
        b.pop_back();
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> buckets_;
  std::vector<std::mutex> locks_;

  inline size_t bucket_index_(uint64_t key) const {
    const uint64_t h = mix64(key);
    return static_cast<size_t>(h % buckets_.size());
  }

  inline std::mutex &lock_for_bucket_(size_t bucket_idx) {
    return locks_[bucket_idx % locks_.size()];
  }
};

