构建命令：
mkdir build
cd build
cmake ../  -DCMAKE_CXX_COMPILER=nvcc -DCMAKE_C_COMPILER=nvcc
make
./uptk_test
