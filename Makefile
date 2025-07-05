hello:
	echo "Hello, World"

benchmark:
	time ./a.out ingredients 1 1 1 1 1 100 10

build:
	nvcc match.cu

test_input: test/test_input.cpp src/input.cpp test/test_ingredients.txt
	g++ -I src -o build/test_input test/test_input.cpp src/input.cpp
	./build/test_input

test_match:
	nvcc -I src -o build/test_match test/test_match.cpp src/input.cpp src/match.cu
	./build/test_match