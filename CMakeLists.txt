cmake_minimum_required(VERSION 3.10)
project(cuda_apis_test LANGUAGES C CXX)

find_package(CUDA REQUIRED)
message(STATUS "ROCM_PATH = $ENV{ROCM_PATH}")

# 头文件路径
include_directories($ENV{ROCM_PATH}/cuda/cuda-11/include)
include_directories($ENV{ROCM_PATH}/include)

# 库文件路径（最关键！）
link_directories($ENV{ROCM_PATH}/lib64)
link_directories($ENV{ROCM_PATH}/cuda/cuda-11/lib64)

# 全局链接所有需要的库（子目录全部能继承）
#target_link_libraries(UPTKrt cudart cuda pthread rt)
#target_link_libraries(cudart cuda pthread rt)

enable_testing()

# 子目录编译（内部会自己生成可执行文件）
add_subdirectory(cuda_test)
add_subdirectory(cublas_test)
add_subdirectory(cufft_test)
add_subdirectory(nccl_test)
