set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{ROCM_PATH}/cuda/cuda-11/include")
add_compile_definitions(CUDA_VERSION=11000)

set(RAND_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
include_directories(${RAND_INCLUDE})

set(_CUDA_LIBDIR "$ENV{ROCM_PATH}/cuda/cuda-11/lib64")
link_directories(${_CUDA_LIBDIR})

file(GLOB RAND_TEST_SOURCES "test/test_UPTKrand*.cpp")

foreach(test_src ${RAND_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
        ${_CUDA_LIBDIR}/libcudart.so
        ${_CUDA_LIBDIR}/libcurand.so
        UPTKrand
        rt
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
