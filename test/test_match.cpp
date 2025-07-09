#include <cassert>
#include <climits>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

const int LARGE_NUMBER = INT_MAX / 2;

// Project the higher dimensional state down to one dimensional index
static int calc_idx (int num_elem, vector<int> ingredient, vector<int> target) {
    int acc = 1;
    int tmp = 0;
    for (int i = 0; i < num_elem; i++) {
        tmp += acc * ingredient[i];
        acc *= (target[i] + 1);
    }
    return tmp;
}

// Project the one dim index back up to the higher dimensional state
static vector<int> idx2elem(int num_elem, vector<int> target, int idx) {
    vector<int> elements(num_elem);
    for (int i = 0; i < num_elem; i++) {
        elements[i] = idx % (target[i] + 1);
        idx /= (target[i] + 1);
    }
    return elements;
}


static vector<vector<int>> calc_dp(int num_elem, int ing_limit, vector<vector<int>> ings, vector<int> target) {
    int n = ings.size();
    int state_size = calc_idx(num_elem, target, target) + 1;
    vector<vector<int>> dp(n + 1, vector<int>(state_size));
    // set initial conditions
    dp[0][0] = 0;
    for (int i = 1; i < dp[0].size(); i++) {
        dp[0][i] = LARGE_NUMBER;
    }

    for (int idx = 1; idx < dp.size(); idx++) {
        vector<int> ing = ings[idx - 1];
        for (int state = 0; state < state_size; state++) {
            vector<int> elems = idx2elem(num_elem, target, state);

            // option 1: don't use current ingredient
            dp[idx][state] = dp[idx-1][state];
            for (int uses = 1; uses <= ing_limit; uses++) {
                // compute previous state
                vector<int> prev_elems = elems;
                for (int i = 0; i < num_elem; i++) {
                    prev_elems[i] -= ing[i] * uses;
                }

                // check uses * ing doesn't overflow
                bool valid = true;
                for (int i : prev_elems) {
                    if (i < 0) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) {
                    break;
                }

                // try using current ingredient
                int prev_idx = calc_idx(num_elem, prev_elems, target);
                dp[idx][state] = min(dp[idx][state], dp[idx - 1][prev_idx] + uses);
            }
        }
    }
    return dp;
}


vector<vector<int>> get_recipe(int num_elem, vector<vector<int>>dp, vector<vector<int>> ingredients, vector<int> target, int max_ings_num) {
    int n = dp.size();
    int idx = calc_idx(num_elem, target, target);
    if (dp[n-1][idx] > max_ings_num) {
        printf("No valid recipe found");
        return vector<vector<int>>(0);
    }

    vector<vector<int>> sol;
    for (int col = n - 1; col >= 0; col--) {
        if (idx == 0) {
            break;
        }

        assert(col > 0);
        vector<int> ing = ingredients[col - 1];
        int ing_idx = calc_idx(num_elem, ing, target);

        for (int uses = 0; uses <= max_ings_num; uses++) {
            if (dp[col - 1][idx - uses * ing_idx] == dp[col][idx] - uses && dp[col-1][idx - uses * ing_idx] != LARGE_NUMBER) {
                idx -= uses * ing_idx;
                for (int i = 0; i < uses; i++) {
                    sol.push_back(ing);
                }
                break;
            }
        }
    }
    assert(idx == 0);
    return sol;
}



int main() {

    // test parameters
    int num_elements = 2;
    int ingredient_limit = 5;
    vector<vector<int>> ingredients = {
        {0, 1},
        {1, 0},
        // {5, 0}
    };
    vector<int> target = {2, 2};

    // run code
    vector<vector<int>> result = calc_dp(num_elements, ingredient_limit, ingredients, target);

    // check values
    vector<vector<int>> exp = {
        {0, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER, LARGE_NUMBER},
        {0, LARGE_NUMBER, LARGE_NUMBER, 1, LARGE_NUMBER, LARGE_NUMBER, 2, LARGE_NUMBER, LARGE_NUMBER},
        {0, 1, 2, 1, 2, 3, 2, 3, 4},
    };

    if (equal(result.begin(), result.end(), exp.begin(), exp.end())) {
        cout << "The DP is correct" << endl;
    }
    else {
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

    vector<vector<int>> recipe = get_recipe(num_elements, result, ingredients, target, ingredient_limit);
    printf("recipe: \n");
    for (auto i : recipe) {
        for (auto j : i) {
            printf("%d ", j);
        }
        printf("\n");
    }

    return 0;
}
