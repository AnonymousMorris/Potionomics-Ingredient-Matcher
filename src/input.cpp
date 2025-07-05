#include "./input.h"
#include "./Ingredient.h"
#include <bits/stdc++.h>

using namespace std;

const int NUMBER_OF_MAGNIUMS = 5;

std::vector<Ingredient> parse_ingredients(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "unable to open ingredients file " << filename << std::endl;
        return vector<Ingredient>();
    }

    vector<Ingredient> ingredients;

    int line_cnt = 0;
    std::string line;
    for(int line_count = 0; getline(file, line); line_count++) {
        std::stringstream ss(line);

        Ingredient ingredient;
        for (int i = 0; i < NUMBER_OF_MAGNIUMS; i++) {
            if (!(ss >> ingredient[i])) {
                cerr << "not enough values on line " << line_count + 1 << "\n";
                break;
            }
            if (i == NUMBER_OF_MAGNIUMS - 1) {
                ingredients.push_back(ingredient);
            }
        }
    }

    sort(ingredients.begin(), ingredients.end());
    ingredients.resize(unique(ingredients.begin(), ingredients.end()) - ingredients.begin());

    return ingredients;
}