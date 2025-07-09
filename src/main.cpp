#include "Ingredient.h"


array<int, 5> calc_max_mags(int caldron_magnum_limit, array<int, 5> recipe_ratios) {
    int sum = std::accumulate(recipe_ratios.begin(), recipe_ratios.end(), 0);
    array<int, 5> mx_magnums;
    for (int i = 0; i < 5; i++) {
        mx_magnums[i] = caldron_magnum_limit * recipe_ratios[i] / sum;
    }
    return mx_magnums;
}


int main() {
    // checks if filename a b c d e mx_magnim mx_ingredients are given

    if (argc < 9) {
        std::cerr << "not enough parameters given" << std::endl;
    }

    // Take in arguments
    vector<Ingredient> parse_ingredients(args[1]);
    int cauldron_mag_limit = stoi(args[2]);
    int cauldron_ingredient_limit = stoi(args[3]);

    // Calculate parameters
    for (int i = 0; i < 5; i++) {
        recipe_ratio[i] = stoi(args[4 + i]);
    }
    array<int, 5> mx_mags = calc_max_mags(cauldron_magnum_limit, parse_ingredients);

    // Run the dynamic programming on GPU
    cuda_calculate_recipe(ingredients, cauldron_ingredient_limit, mx_mags);

    return 0;
}
