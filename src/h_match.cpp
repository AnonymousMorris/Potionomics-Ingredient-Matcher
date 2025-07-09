#include "./input.h"

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>
#include <bits/stdc++.h>
#include <bits/types/stack_t.h>


using namespace std;

const string filename = "../ingredients";
const int cauldron_mag_limit = 5;
const int cauldron_ing_limit = 3;
const array<int, 5> recipe_ratio = {1, 0, 0, 0, 0};

const int LARGE_NUMBER = INT_MAX / 2;

static array<int, 5> calc_max_mags(int caldron_magnum_limit, array<int, 5> recipe_ratios) {
    int sum = std::accumulate(recipe_ratios.begin(), recipe_ratios.end(), 0);
    array<int, 5> mx_mags;
    for (int i = 0; i < 5; i++) {
        mx_mags[i] = caldron_magnum_limit * recipe_ratios[i] / sum;
    }
    return mx_mags;
}


// calculate an int representing the ingredient state (we're flattening the 5d array into 1d)
static int calc_index(Ingredient ingredient) {
    int state = 0;
    int accumulate = 1;
    for (int i = 4; i >= 0; i--) {
        state += ingredient[i] * accumulate;
        accumulate *= ingredient[i] + 1;
    }
    return state;
}


// NOTE: We are adding a blank first column to simplify the logic (ie. There is num of ingredients + 1 columns)
void calc_dp(int* dp, Ingredient* ingredients, int state_size, array<int, 5> cauldron_max_mags, int ing_num) {

    for (int ing_idx = 0; ing_idx < ing_num; ing_idx++) {
        Ingredient ingredient = ingredients[ing_idx];

        int ingredient_state = calc_index(ingredient);


        int prev_offset = ing_idx * state_size;
        int offset = prev_offset + state_size;
        
        // dp loop for each column
        array<int, 5> state;
        for (state[0] = 0; state[0] <= cauldron_max_mags[0]; state[0]++) {
            for (state[1] = 0; state[1] <= cauldron_max_mags[1]; state[1]++) {
                for (state[2] = 0; state[2] <= cauldron_max_mags[2]; state[2]++) {
                    for (state[3] = 0; state[3] <= cauldron_max_mags[3]; state[3]++) {
                        for (state[4] = 0; state[4] <= cauldron_max_mags[4]; state[4]++) {

                            int idx = calc_index(state);
                            // option 1: don't use this ingredient
                            dp[offset + idx] = dp[prev_offset + idx];
                            
                            // option 2: use this ingredient 1 or more times
                            for (int uses = 1; uses <= cauldron_ing_limit; uses++) {
                                // check if current uses of ingredient is greater than current state under consideration
                                array<int, 5> prev_state;
                                bool valid = true;

                                // calculate prev state and validate
                                for (int i = 0; i < 5; i++) {
                                    prev_state[i] = state[i] - uses * ingredient[i];
                                    if (prev_state[i] < 0) {
                                        valid = false;
                                    }
                                }

                                // update dp if valid
                                if (valid) {
                                    int prev_idx = calc_index(prev_state);
                                    dp[offset + idx] = min(dp[offset + idx], dp[prev_offset + prev_idx] + uses);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// NOTE: The solution is returned through C style pointer
// The return value is the number of ingredients used in the solution
int get_recipe(int* dp, Ingredient* ingredients, array<int, 5> max_mags, int state_size, int ing_num, int* sol) {

    array<int, 5> cur_state = max_mags;
    int cur_idx = calc_index(cur_state);
    int col = ing_num; // `ingredients.size() - 1`
    int ingredients_used;

    // check if an answer exists
    ingredients_used = dp[col * state_size + cur_idx];
    if (ingredients_used > cauldron_ing_limit) {
        printf("No Answer Found");
        return 0;
    }

    sol = (int*) malloc(cauldron_ing_limit * sizeof(int));
    int sol_idx = 0;
    // backtrack one answer
    for (; col > 0; col--) {
        if (cur_idx == 0) {
            break;
        }

        int prev_col = col - 1;
        int ingredient_idx = calc_index(ingredients[col - 1]);
        for (int i = 0; i <= cauldron_ing_limit; i++) {
            if (dp[prev_col * state_size + ingredient_idx] == dp[col * state_size + cur_idx] - i) {
                sol[sol_idx++] = col - 1;
                cur_idx -= i * ingredient_idx;
            }
        }
    }

    return ingredients_used;
}

void match() {
    vector<Ingredient> ingredients = parse_ingredients(filename);
    int ing_num = ingredients.size();
    array<int, 5> max_mags = calc_max_mags(cauldron_mag_limit, recipe_ratio);

    // calculate state size
    int state_size = 1;
    for (int i : max_mags) {
        state_size *= (i + 1);
    }

    int* dp = (int*) malloc(state_size * (ing_num + 1) * sizeof(int));
    for (int i = 0; i < state_size * (ing_num + 1); i++) {
        dp[i] = LARGE_NUMBER;
    }
    dp[0] = 0;

    calc_dp(dp, ingredients.data(), state_size, max_mags, ing_num);


    // printing dp table in the correct orientation
    for (int i = ing_num; i >= 0; i--) {
        for (int state = state_size - 1; state >= 0; state--) {
            int idx = i * state_size + state;
            printf("%d ", dp[idx]);
        }
        printf("\n");
    }

    int* sol;
    int sol_n;
    sol_n = get_recipe(dp, ingredients.data(), max_mags, state_size, ing_num, sol);

    for (int i = 0; i < sol_n; i++) {
        printf("%d ", sol[i]);
    }
}

int main() {
    match();
}
