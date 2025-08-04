#include <cuda.h>

#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_KERNEL_CHECK(kernel_name, grid, block, ...)                                           \
    do {                                                                                           \
        printf("Launching kernel: %s with grid(%d,%d,%d) block(%d,%d,%d)\n", #kernel_name, grid.x, \
               grid.y, grid.z, block.x, block.y, block.z);                                         \
        kernel_name<<<grid, block>>>(__VA_ARGS__);                                                 \
        cudaError_t launch_error = cudaGetLastError();                                             \
        if (launch_error != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA kernel launch error for %s at %s:%d - %s\n", #kernel_name,       \
                    __FILE__, __LINE__, cudaGetErrorString(launch_error));                         \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        cudaError_t sync_error = cudaDeviceSynchronize();                                          \
        if (sync_error != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA kernel execution error for %s at %s:%d - %s\n", #kernel_name,    \
                    __FILE__, __LINE__, cudaGetErrorString(sync_error));                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        printf("Kernel %s completed successfully\n", #kernel_name);                                \
    } while (0)

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

// @ return the number of ingredients used in the result
__global__ void get_recipe_kernel(int n, int* dp, int* ingredients, int* target, int max_ings_num,
                                  int* sol_n, int* sol) {
    // n = dp.size();
    *sol_n = 0;
    int idx, state_size, cur_offset, prev_offset;
    idx = calc_idx(target, target);
    state_size = idx + 1;
    cur_offset = (n - 1) * state_size;
    // printf("offset: %d, idx: %d, dp val: %d, max_ing: %d\n", cur_offset, idx, dp[cur_offset +
    // idx],
    //        max_ings_num);
    if (dp[cur_offset + idx] > max_ings_num) {
        printf("No valid recipe found\n");
        return;
    }

    int* ing;
    int ing_idx, uses;
    // printf("n: %d, idx: %d\n", n, idx);
    for (int col = n - 1; col > 0; col--) {
        // found recipe
        if (idx == 0) {
            uses = 0;
            break;
        }

        ing = ingredients + ((col - 1) * NUM_ELEM);
        ing_idx = calc_idx(ing, target);

        for (uses = 0; uses <= max_ings_num; uses++) {
            prev_offset = (col - 1) * state_size;
            cur_offset = col * state_size;
            int dp_prev = dp[prev_offset + idx - uses * ing_idx];
            int dp_cur = dp[cur_offset + idx];
            assert(0 <= prev_offset && prev_offset < state_size * n);
            assert(0 <= cur_offset && cur_offset < state_size * n);
            if (dp_prev == dp_cur - uses) {
                assert(dp[prev_offset + idx - uses * ing_idx] != LARGE_NUMBER);
                idx -= uses * ing_idx;
                for (int i = 0; i < uses; i++) {
                    for (int j = 0; j < NUM_ELEM; j++) {
                        int sol_offset = *sol_n * NUM_ELEM;
                        sol[sol_offset + j] = ing[j];
                    }
                    *sol_n += 1;
                }
                break;
            }
        }
    }
    for (int i = 0; i < *sol_n * NUM_ELEM; i++) {
        printf("%d ", sol[i]);
    }
    printf("\n");
}

__host__ vector<vector<int>> get_recipe_host(int n, int ing_limit, vector<vector<int>> dp,
                                             int* h_ings, int* target) {
    vector<int> h_dp;
    for (auto dp_row : dp) {
        for (auto dp_col : dp_row) {
            h_dp.push_back(dp_col);
        }
    }
    int state_size = calc_idx(target, target) + 1;
    int h_result_n;
    int *d_dp, *d_ings, *d_result_n, *d_result, *d_target;
    // h_dp = (int*)malloc(n * state_size * sizeof(int));

    CUDA_CHECK(cudaMalloc((void**)&d_dp, (n + 1) * state_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ings, n * NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, ing_limit * NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_target, NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_result_n, sizeof(int)));

    CUDA_CHECK(
        cudaMemcpy(d_dp, h_dp.data(), (n + 1) * state_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ings, h_ings, n * NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_KERNEL_CHECK(get_recipe_kernel, dim3(1), dim3(1), n, d_dp, d_ings, d_target, ing_limit,
                      d_result_n, d_result);
    CUDA_CHECK(cudaMemcpy(&h_result_n, d_result_n, sizeof(int), cudaMemcpyDeviceToHost));
    printf("result_n: %d\n", h_result_n);
    vector<int> h_result(h_result_n * NUM_ELEM);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, h_result_n * NUM_ELEM * sizeof(int),
                          cudaMemcpyDeviceToHost));
    vector<vector<int>> ans(h_result_n, vector<int>(NUM_ELEM));
    for (int i = 0; i < h_result_n; i++) {
        for (int j = 0; j < NUM_ELEM; j++) {
            ans[i][j] = h_result[i * NUM_ELEM + j];
        }
    }
    return ans;
}
int main() {
    bool test_passed = true;

    // test parameters
    int cauldron_ings_limit = 5;
    int num_ingredients = 2;
    int ingredients[] = {
        0,
        1,
        1,
        0,
    };

    vector<int> target = {2, 2};

    // run code
    vector<vector<int>> result =
        calc_dp(cauldron_ings_limit, num_ingredients, ingredients, target.data());

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
        test_passed = false;

        cout << "The DP is incorrect" << endl;

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
    }

    // Check that we can correctly extract a recipe out of the dynamic programming
    vector<vector<int>> recipe = get_recipe_host(num_ingredients + 1, cauldron_ings_limit, result,
                                                 ingredients, target.data());
    vector<int> total(NUM_ELEM, 0);
    for (auto mags : recipe) {
        for (int idx = 0; idx < mags.size(); idx++) {
            total[idx] += mags[idx];
        }
    }
    if (total == target) {
        printf("The Recipe is correct\n");
    } else {
        test_passed = false;
        printf("Recipe is wrong, ingredients: ");
        for (auto mags : recipe) {
            for (int amt : mags) {
                printf("%d ", amt);
            }
            printf("\n");
        }
    }

    // Print result for clarity
    if (test_passed) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }

    return test_passed ? 0 : 1;
}
