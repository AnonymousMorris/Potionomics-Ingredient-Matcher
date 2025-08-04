hello:
	echo "Hello, World"

benchmark:
	time ./a.out ingredients 1 1 1 1 1 100 10

# build:
# 	nvcc match.cu

# test_input: test/test_input.cpp src/input.cpp test/test_ingredients.txt
# 	g++ -I src -o build/test_input test/test_input.cpp src/input.cpp
# 	./build/test_input

./build/test_gpu: ./test/test_gpu.cu
	nvcc ./test/test_gpu.cu -o ./build/test_gpu

test_gpu: ./build/test_gpu
	./build/test_gpu

./build/test_cpu: ./test/test_cpu.cpp
	nvcc ./test/test_cpu.cpp -o ./build/test_cpu

test_cpu: ./build/test_cpu
	./build/test_cpu