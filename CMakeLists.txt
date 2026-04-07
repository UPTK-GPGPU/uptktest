cmake_minimum_required(VERSION 3.10.0)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})

# set(GTEST_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest-build)
set(GTEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/googletest)
# set(GTEST_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/googletest-install)
# file(MAKE_DIRECTORY ${GTEST_BUILD})
# file(MAKE_DIRECTORY ${GTEST_INSTALL})

project(cudatest)
find_package(CUDA)

# find_package(GTest)
add_compile_options(-xcu)
enable_testing()
include_directories(${GTEST_SRC}/include ${GTEST_SRC})
add_subdirectory(googletest)

if (NOT DEFINED ROCM_PATH)
	set(ROCM_PATH $ENV{ROCM_PATH})
endif ()

set (LLVM_PATH "${ROCM_PATH}/llvm"  CACHE STRING "Default LLVM installation directory." )

message (STATUS "ROCM_PATH: ${ROCM_PATH}")
# message (STATUS "llvm path: ${LLVM_PATH}")

# # compiler googletest at configure time
# execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_COMPILER=${LLVM_PATH}/bin/clang++ -DCMAKE_C_COMPILER=${LLVM_PATH}/bin/clang -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL} "${CMAKE_GENERATOR}" ${GTEST_SRC}
# 	       RESULT_VARIABLE result
# 	       WORKING_DIRECTORY ${GTEST_BUILD})
# if(result)
#    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
# endif()

# execute_process(COMMAND make 
# 	       RESULT_VARIABLE result
# 	       WORKING_DIRECTORY ${GTEST_BUILD})
# if(result)
#    message(FATAL_ERROR "make step for googletest failed: ${result}")
# endif()

# execute_process(COMMAND make install
# 	       RESULT_VARIABLE result
# 	       WORKING_DIRECTORY ${GTEST_BUILD})
# if(result)
#    message(FATAL_ERROR "make install step for googletest failed: ${result}")
# endif()


#set(CMAKE_CXX_COMPILER "${ROCM_PATH}/cuda/bin/nvcc")
#set(CMAKE_C_COMPILER "${LLVM_PATH}/bin/clang")

# include_directories("${GTEST_INSTALL}/include")

# Now simply link against gtest or gtest_main as needed. Eg
set(TEST_SOURCE_FILES "")
add_subdirectory(src)
cuda_add_executable(uptk_test ${TEST_SOURCE_FILES})

# if (EXISTS "${GTEST_INSTALL}/lib64/libgtest.a")
#    target_link_libraries(cudamocker_test ${GTEST_INSTALL}/lib64/libgtest.a)
# else()
#    target_link_libraries(cudamocker_test ${GTEST_INSTALL}/lib/libgtest.a)
# endif()


target_link_libraries(uptk_test pthread gtest UPTKfft UPTKblas UPTKrt UPTKnccl)


