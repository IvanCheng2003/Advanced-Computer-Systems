#pragma once
#include <vector>

struct CSR {
    int m = 0;  // rows
    int k = 0;  // cols
    std::vector<int>   row_ptr;  // size m+1
    std::vector<int>   col_idx;  // size nnz
    std::vector<float> values;   // size nnz
};
