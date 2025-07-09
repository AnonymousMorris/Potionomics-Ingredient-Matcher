// #pragma GCC optimize("O3","unroll-loops")
// #pragma GCC target("avx","avx2")
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Ingredient.h"

using namespace std;

/* We store ingredient magniums in ingredient_offsets. The first ingredient takes up
ingredient_offsets[0..5). The second one takes up ingredient_offsets[5..10), etc. */
__constant__ int ingredient_offsets[1024];

__global__ void calc_dp(int* dp, int i, int max_ingredients, int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) {
        return;
    }
    int offset = ingredient_offsets[i];
    int tmp = dp[idx + (i - 1) * n_states];
    for (int ii = 1; ii <= max_ingredients; ii++) {
        int prev = idx - offset * ii;
        if (prev < 0) {
            break;
        }
        if (dp[prev + (i - 1) * n_states] > 0 || prev == 0) {
            tmp = min(tmp, dp[prev + (i - 1) * n_states] + ii);
        }
    }
    dp[idx + i * n_states] = tmp;
}

__global__ void calc_dp_first(int* dp, int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) {
        return;
    }
    int offset = ingredient_offsets[0];
    if (idx % offset == 0) {
        int dp_idx = idx / offset;
        dp[idx] = dp_idx;
    }
}

__global__ void get_recipe(int* dp, int n_ingredients, int n_states, int* result, int* result_n) {
    result_n = 0;
    int idx = n_ingredients * n_states - 1;
    for (int i = n_ingredients - 1; i > 0; i--) {
        int ing_offset = ingredient_offsets[i];
        if (ing_offset <= idx % n_states) {
            if (dp[idx - ing_offset] == dp[idx] - 1) {
                result[*result_n++] = i;
            }
        }
    }
    if (idx == ingredient_offsets[0]) {
        result[*result_n++] = 0;
    }
}

int calc_index(Ingredient ingredient, Ingredient mx_mags) {
    int val = 0;
    for (int i = 0, base = 1; i < 5; i++) {
        val += base * ingredient[i];
        base *= (mx_mags[i] + 1);
    }
    return val;
}

void cuda_calculate_recipe(
    vector<Ingredient> ingredients,  // a vector of ingredients
    int ingredient_limit,            // the maximum number of ingredients allowed
    array<int, 5> maximum_magniums   // maximum maginums the dp will try to get
) {
    int max_states = 1;
    for (int i : maximum_magniums) {
        max_states *= i;
    }
    int num_ingredients = ingredients.size();
    assert(num_ingredients < 1024);

    // Need to prepare ingredient data for copying to constant memory
    // This assumes you have a way to extract offsets from ingredients
    int h_ingredient_offsets[1024];
    // Fill h_ingredient_offsets with data from ingredients vector
    // ... (add your logic here)

    cudaMemcpyToSymbol(ingredient_offsets, h_ingredient_offsets, sizeof(int) * num_ingredients);

    int* dp;
    int size = num_ingredients * max_states;
    cudaMalloc((void**)&dp, sizeof(int) * size);
    cudaMemset(dp, 0, sizeof(int) * size);

    calc_dp_first<<<dim3((max_states + 1023) / 1024, 1, 1), dim3(1024, 1, 1)>>>(dp, max_states);

    for (int i = 1; i < num_ingredients; i++) {
        calc_dp<<<dim3((max_states + 1023) / 1024, 1, 1), dim3(1024, 1, 1)>>>(
            dp, i, ingredient_limit, max_states);
    }

    int* h_result = (int*)malloc(num_ingredients * sizeof(int));
    int h_result_n;

    int* d_result;
    int* d_result_n;
    cudaMalloc((void**)&d_result, num_ingredients * sizeof(int));
    cudaMalloc((void**)&d_result_n, sizeof(int));
    get_recipe<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(dp, num_ingredients, max_states, d_result,
                                                 d_result_n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, &d_result, num_ingredients, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_n, &d_result_n, 1, cudaMemcpyDeviceToHost);

    cudaFree(dp);
}
