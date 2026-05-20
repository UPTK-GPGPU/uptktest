set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{ROCM_PATH}/cuda/cuda-11/include")
add_compile_definitions(CUDA_VERSION=11000)

set(DRIVER_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
include_directories(${DRIVER_INCLUDE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(_CUDA_LIBDIR "$ENV{ROCM_PATH}/cuda/cuda-11/lib64")
link_directories(${_CUDA_LIBDIR}/stubs ${_CUDA_LIBDIR})
if(EXISTS "${_CUDA_LIBDIR}/stubs/libcuda.so")
    set(CUDA_DRIVER_SO "${_CUDA_LIBDIR}/stubs/libcuda.so")
elseif(EXISTS "${_CUDA_LIBDIR}/libcuda.so")
    set(CUDA_DRIVER_SO "${_CUDA_LIBDIR}/libcuda.so")
else()
    set(CUDA_DRIVER_SO "cuda")
endif()

file(GLOB DRIVER_TEST_SOURCES "test/test_UP*.cpp")

foreach(test_src ${DRIVER_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
        $ENV{ROCM_PATH}/cuda/cuda-11/lib64/libcudart.so
        ${CUDA_DRIVER_SO}
        UPTKdrt
        rt
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
