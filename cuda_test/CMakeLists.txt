set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I$ENV{ROCM_PATH}/cuda/cuda-11/include")

file(GLOB TEST_SOURCES "test/test_*.cpp" "test/test_*.cu")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    get_filename_component(test_ext ${test_src} EXT)
    if(test_ext STREQUAL ".cu")
        # For .cu files, use nvcc directly with custom command
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{ROCM_PATH}/lib:$ENV{ROCM_PATH}/cuda/cuda-11/lib64:$ENV{LD_LIBRARY_PATH}
                    $ENV{ROCM_PATH}/cuda/cuda-11/bin/nvcc ${test_src} -o ${output_file}
                    -I$ENV{ROCM_PATH}/cuda/cuda-11/include
                    -L$ENV{ROCM_PATH}/cuda/cuda-11/lib64 -lcudart -lUPTKrt -lrt
                    -Wl,-rpath,$ENV{ROCM_PATH}/cuda/cuda-11/lib64 -Wl,-rpath,/opt/dtk/hip/lib
            DEPENDS ${test_src}
            COMMENT "Building CUDA test ${test_name}"
        )
        link_libraries(rt)
        add_custom_target(${test_name} ALL DEPENDS ${output_file})
    else()
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} $ENV{ROCM_PATH}/cuda/cuda-11/lib64/libcudart.so UPTKrt rt)
        link_libraries(rt)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
