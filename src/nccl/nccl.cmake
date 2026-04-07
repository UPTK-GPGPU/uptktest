message(STATUS "include nccl test case")

set(NCCL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl)

set(NCCL_TEST_FILES
    ${NCCL_DIR}/UPTKncclGetVersion.cu
    ${NCCL_DIR}/UPTKncclGetUniqueId.cu
    ${NCCL_DIR}/UPTKncclCommInitRank.cu
    ${NCCL_DIR}/UPTKncclCommInitAll.cu
    ${NCCL_DIR}/UPTKncclCommDestroy.cu
    ${NCCL_DIR}/UPTKncclCommAbort.cu
    ${NCCL_DIR}/UPTKncclGetErrorString.cu
    ${NCCL_DIR}/UPTKncclCommGetAsyncError.cu
    ${NCCL_DIR}/UPTKncclCommCount.cu
    ${NCCL_DIR}/UPTKncclCommCuDevice.cu
    ${NCCL_DIR}/UPTKncclCommUserRank.cu
    ${NCCL_DIR}/UPTKncclReduce.cu
    ${NCCL_DIR}/UPTKncclBroadcast.cu
    ${NCCL_DIR}/UPTKncclAllReduce.cu
    ${NCCL_DIR}/UPTKncclReduceScatter.cu
    ${NCCL_DIR}/UPTKncclSend.cu
    ${NCCL_DIR}/UPTKncclRecv.cu
    ${NCCL_DIR}/UPTKncclGroupStart.cu
    ${NCCL_DIR}/UPTKncclGroupEnd.cu
    ${NCCL_TEST_TMP_FILES}
    )