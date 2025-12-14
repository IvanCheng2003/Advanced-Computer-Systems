#include "matrix_gen.hpp"
#include <random>

std::vector<float> make_dense_random(int rows, int cols,
                                     float min_val,
                                     float max_val,
                                     unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> M(rows * cols);
    for (auto &x : M) x = dist(rng);
    return M;
}

CSR generate_csr_random(int m, int k,
                        double density,
                        float min_val,
                        float max_val,
                        unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> val_dist(min_val, max_val);
    std::bernoulli_distribution nz_dist(density);

    CSR A;
    A.m = m;
    A.k = k;
    A.row_ptr.resize(m + 1);

    std::vector<int>   cols;
    std::vector<float> vals;
    int nnz = 0;

    for (int i = 0; i < m; ++i) {
        A.row_ptr[i] = nnz;
        for (int j = 0; j < k; ++j) {
            if (nz_dist(rng)) {
                cols.push_back(j);
                vals.push_back(val_dist(rng));
                ++nnz;
            }
        }
    }
    A.row_ptr[m] = nnz;
    A.col_idx = std::move(cols);
    A.values  = std::move(vals);
    return A;
}
