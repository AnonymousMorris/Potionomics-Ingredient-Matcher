#pragma once
#include <vector>
#include <string>
#include "./Ingredient.h"

std::vector<Ingredient> parse_ingredients(const std::string& filename);