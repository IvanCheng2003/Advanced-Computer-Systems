#pragma once
#include <vector>
#include "csr.hpp"

// Dense random matrix (row-major, size rows x cols)
std::vector<float> make_dense_random(int rows, int cols,
                                     float min_val = -1.0f,
                                     float max_val = 1.0f,
                                     unsigned long long seed = 0);

// Generate a sparse matrix directly into CSR form with Bernoulli(density)
CSR generate_csr_random(int m, int k,
                        double density,
                        float min_val = -1.0f,
                        float max_val = 1.0f,
                        unsigned long long seed = 0);
