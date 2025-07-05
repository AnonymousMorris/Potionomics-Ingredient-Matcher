#include "../src/match.h"

#include <iostream>
using namespace std;

int main() {
    vector<Ingredient> ingredients = {
        {1, 2, 3, 4, 5},
        {0, 1, 0, 2, 1},
        {5, 0, 3, 1, 2}
    };
    int ingredient_limit = 5;
    std::array<int, 5> maximum_magniums  = { 100, 100, 100, 100, 100 };
    cuda_calculate_recipe(ingredients, ingredient_limit, maximum_magniums);

    cout << "Hello World!" << endl;
    return 0;
}
