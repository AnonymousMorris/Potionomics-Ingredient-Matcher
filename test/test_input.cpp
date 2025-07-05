#include "../src/input.h"

#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

// Template function to print arrays with << operator
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    os << "[";
    for (size_t i = 0; i < N; ++i) {
        os << arr[i];
        if (i < N - 1) os << ", ";
    }
    os << "]";
    return os;
}

int main() {
    vector<Ingredient> ingredients = parse_ingredients("./test/test_ingredients.txt");

    vector<Ingredient> correct_ingredients = {
        {1, 2, 3, 4, 5},
        {0, 1, 0, 2, 1},
        {5, 0, 3, 1, 2}
    };

    sort(correct_ingredients.begin(), correct_ingredients.end());
    correct_ingredients.resize(unique(correct_ingredients.begin(), correct_ingredients.end()) - correct_ingredients.begin());

    if (ingredients.size() != correct_ingredients.size()) {
        cout << "Error: Ingredients not the same size" << endl;
        return 1;
    }

    for (int i = 0; i < ingredients.size(); i++) {
        if (ingredients[i] != correct_ingredients[i]) {
            cout << "Error: index " << i << " did not match:" << "\n";
            cout << "Parsed ingredients: " << ingredients[i] << "\n";
            cout << "Correct ingredients: " << correct_ingredients[i] << endl;
            return 1;
        }
    }

    cout << "Test passed" << endl;
    return 0;
}