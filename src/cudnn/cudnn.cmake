message(STATUS "include cudnn test case")
set(CUDNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cudnn)

set(CUDNN_TEST_FILES
    ${CUDNN_DIR}/cudnn_activation.cu
    ${CUDNN_DIR}/cudnn_binary.cu
    ${CUDNN_DIR}/cudnn_fill.cu
    ${CUDNN_DIR}/cudnn_memory.cu
    ${CUDNN_DIR}/cudnn_pooling.cu
    ${CUDNN_DIR}/cudnn_reduction.cu
    ${CUDNN_DIR}/cudnn_scale.cu
    ${CUDNN_DIR}/cudnn_softmax.cu
    ${CUDNN_TEST_TMP_FILES}
    )

