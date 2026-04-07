message(STATUS "include graph test case")
set(RUNTIME_GRAPH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/graph)

set(RUNTIME_GRAPH_TEST_FILES
        # cuda11.0 #${RUNTIME_GRAPH_DIR}/cudaGraphTest.cu
        # hip test failed #${RUNTIME_GRAPH_DIR}/cudaGraphTest_1.cu
        # hip test failed #${RUNTIME_GRAPH_DIR}/cudaGraphTest_2.cu
)
