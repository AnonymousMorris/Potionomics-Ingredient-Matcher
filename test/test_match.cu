#include <cuda.h>

#include <cassert>
#include <climits>
#include <cstdio>
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

__host__ vector<int> get_recipe_host(int n, int ing_limit, int* h_dp, int* h_ings, int* target) {
    int state_size = calc_idx(target, target) + 1;
    int h_result_n;
    int *d_dp, *d_ings, *d_result_n, *d_result, *d_target;
    // h_dp = (int*)malloc(n * state_size * sizeof(int));

    CUDA_CHECK(cudaMalloc((void**)&d_dp, (n + 1) * state_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ings, n * NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, ing_limit * NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_target, NUM_ELEM * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_result_n, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_dp, h_dp, (n + 1) * state_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ings, h_ings, n * NUM_ELEM * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_KERNEL_CHECK(get_recipe_kernel, dim3(1), dim3(1), n, d_dp, d_ings, d_target, ing_limit,
                      d_result_n, d_result);
    CUDA_CHECK(cudaMemcpy(&h_result_n, d_result_n, sizeof(int), cudaMemcpyDeviceToHost));
    printf("result_n: %d\n", h_result_n);
    vector<int> h_result(h_result_n * NUM_ELEM);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, h_result_n * NUM_ELEM * sizeof(int),
                          cudaMemcpyDeviceToHost));
    return h_result;
}

int main() {
    // test parameters
    int ingredient_limit = 5;
    int num_ingredients = 2;

    int ingredients[] = {
        0,
        1,
        1,
        0,
    };

    int target[] = {2, 2};

    // check values
    int dp[] = {0,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                LARGE_NUMBER,
                0,
                LARGE_NUMBER,
                LARGE_NUMBER,
                1,
                LARGE_NUMBER,
                LARGE_NUMBER,
                2,
                LARGE_NUMBER,
                LARGE_NUMBER,
                0,
                1,
                2,
                1,
                2,
                3,
                2,
                3,
                4};
    vector<int> ans =
        get_recipe_host(num_ingredients + 1, ingredient_limit, dp, ingredients, target);
    for (int i = 0; i < ans.size() / NUM_ELEM; i++) {
        for (int j = 0; j < NUM_ELEM; j++) {
            cout << ans[i * NUM_ELEM + j] << " ";
        }
        cout << endl;
    }

    return 0;
}