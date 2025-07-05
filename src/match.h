#pragma once

#include <vector>
#include <array>
#include "./Ingredient.h"

void cuda_calculate_recipe(
    std::vector<Ingredient> ingredients, // a vector of ingredients
    int ingredient_limit,                // the maximum number of ingredients allowed
    std::array<int, 5> maximum_magniums  // maximum maginums the dp will try to get
);