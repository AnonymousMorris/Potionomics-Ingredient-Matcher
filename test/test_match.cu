#include <cuda.h>

#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

const int LARGE_NUMBER = INT_MAX / 2;
const int NUM_ELEM = 2;

static int *d_dp, *d_ing, *d_target;

// Project the higher dimensional state down to one dimensional index
__host__ __device__ int calc_idx(int* ingredient, int* target) {
    int acc = 1;
    int tmp = 0;
    for (int i = 0; i < NUM_ELEM; i++) {
        tmp += acc * ingredient[i];
        acc *= (target[i] + 1);
    }
    return tmp;
}

// Project the one dim index back up to the higher dimensional state
__device__ void idx2elem(int* target, int idx, int* result) {
    for (int i = 0; i < NUM_ELEM; i++) {
        result[i] = idx % (target[i] + 1);
        idx /= (target[i] + 1);
    }
}

__global__ void compute_dp_col(int state_space, int idx, int ing_limit, int* ing, int* target,
                               int* dp) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (state_idx < state_space) {
        int offset = idx * state_space;
        int state[NUM_ELEM];
        idx2elem(target, state_idx, state);

        dp[offset + state_idx] = dp[offset - state_space + state_idx];
        for (int uses = 1; uses < ing_limit; uses++) {
            // compute previous state
            for (int i = 0; i < NUM_ELEM; i++) {
                state[i] -= ing[i];
            }

            // check uses * ing doesn't overflow
            bool valid = true;
            for (int i = 0; i < NUM_ELEM; i++) {
                if (state[i] < 0) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                break;
            }

            int prev_idx = calc_idx(state, target);
            dp[offset + state_idx] =
                min(dp[offset + state_idx], dp[offset - state_space + prev_idx] + uses);
        }
    }
}

// @brief sets up kernels
// @param n number of ingredients
__host__ vector<vector<int>> calc_dp(int ing_limit, int n_ing, int* ings, int* target) {
    int state_size = calc_idx(target, target) + 1;
    int n = n_ing + 1;
    int* h_dp;
    // int *d_dp, *d_ing, *d_target;

    h_dp = (int*)malloc(state_size * n * sizeof(int));
    cudaMalloc((void**)&d_dp, state_size * n * sizeof(int));
    cudaMalloc((void**)&d_ing, NUM_ELEM * sizeof(int));
    cudaMalloc((void**)&d_target, NUM_ELEM * sizeof(int));
    assert(h_dp);

    // set up initial values
    cudaMemcpy(d_target, target, NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice);
    // set up first col of dp
    h_dp[0] = 0;
    for (int i = 1; i < state_size; i++) {
        h_dp[i] = LARGE_NUMBER;
    }
    cudaMemcpy(d_dp, h_dp, state_size * sizeof(int), cudaMemcpyHostToDevice);

    for (int idx = 1; idx < n; idx++) {
        int* h_ing = ings + NUM_ELEM * (idx - 1);
        cudaMemcpy(d_ing, h_ing, NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice);
        compute_dp_col<<<(state_size + 1023) / 1024, 1024>>>(state_size, idx, ing_limit, d_ing,
                                                             d_target, d_dp);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_dp, d_dp, state_size * n * sizeof(int), cudaMemcpyDeviceToHost);

    vector<vector<int>> result(n);
    for (int i = 0; i < n; i++) {
        int* h_dp_begin = h_dp + i * state_size;
        result[i] = vector<int>(h_dp_begin, h_dp_begin + state_size);
    }
    free(h_dp);
    return result;
}

// __host__ int get_recipe()

// @ return the number of ingredients used in the result
__device__ int get_recipe(int n, int* dp, int* ingredients, int* target, int max_ings_num,
                          int* result) {
    int idx = calc_idx(target, target);
    int state_size = idx + 1;
    int offset = (n - 1) * state_size;
    int result_n = 0;
    if (dp[offset + idx] > max_ings_num) {
        return -1;
    }

    for (int col = n - 1; col >= 0; col--) {
        if (idx == 0) {
            break;
        }

        int* ing = ingredients + (col * NUM_ELEM);
        int ing_idx = calc_idx(ing, target);

        for (int uses = 0; uses <= max_ings_num; uses++) {
            int offset = col * state_size;
            if (dp[offset - state_size + idx - uses * ing_idx] == dp[offset + idx] - uses &&
                dp[offset - state_size + idx - uses * ing_idx] != LARGE_NUMBER) {
                idx -= uses * ing_idx;
                for (int i = 0; i < uses; i++) {
                    result[result_n++] = col - 1;
                }
            }
        }
    }
    return result_n;
}

int main() {
    // test parameters
    // int num_elements = 2;
    int ingredient_limit = 5;
    int num_ingredients = 2;
    int ingredients[] = {
        0,
        1,
        1,
        0,
    };

    int target[] = {2, 2};

    // run code
    vector<vector<int>> result = calc_dp(ingredient_limit, num_ingredients, ingredients, target);

    // check values
    vector<vector<int>> exp = {
        {0, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER,
         LARGE_NUMBER, LARGE_NUMBER},
        {0, LARGE_NUMBER, LARGE_NUMBER, 1, LARGE_NUMBER, LARGE_NUMBER, 2, LARGE_NUMBER,
         LARGE_NUMBER},
        {0, 1, 2, 1, 2, 3, 2, 3, 4},
    };

    if (equal(result.begin(), result.end(), exp.begin(), exp.end())) {
        cout << "The DP is correct" << endl;
    } else {
        cout << "The DP is incorrect" << endl;
    }

    cout << "Expected: \n";
    for (auto i : exp) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }

    cout << "Actual: \n";
    for (auto i : result) {
        for (auto j : i) {
            cout << j << " ";
        }
        cout << endl;
    }

    // int* h_recipe;
    // int* d_recipe;

    // h_recipe = (int*)malloc(ingredient_limit * sizeof(int));
    // cudaMalloc((void**)&d_recipe, ingredient_limit * sizeof(int));

    //     printf("recipe: \n");
    // for (auto i : recipe) {
    //     for (auto j : i) {
    //         printf("%d ", j);
    //     }
    //     printf("\n");
    // }

    return 0;
}
