message(STATUS "include cusparse test case")

set(CUSPARSE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cusparse)

set(CUSPARSE_TEST_FILES
    ${CUSPARSE_DIR}/coosort_example.cu
    ${CUSPARSE_DIR}/graph_capture_example.cu
    ${CUSPARSE_DIR}/rot_example.cu
    ${CUSPARSE_DIR}/scatter_example.cu
    ${CUSPARSE_DIR}/sddmm_csr_example.cu
    ${CUSPARSE_DIR}/sparse2dense_csr_example.cu
    #${CUSPARSE_DIR}/spgemm_example.cu
    #${CUSPARSE_DIR}/spgemm_mem_example.cu
    #${CUSPARSE_DIR}/spgemm_reuse_example.cu
    #${CUSPARSE_DIR}/spmm_blockedell_example.cu
    ${CUSPARSE_DIR}/spmm_coo_batched_example.cu
    ${CUSPARSE_DIR}/spmm_coo_example.cu
    ${CUSPARSE_DIR}/spmm_csr_batched_example.cu
    ${CUSPARSE_DIR}/spmm_csr_example.cu
    #${CUSPARSE_DIR}/spmm_csr_op_example.cu
    ${CUSPARSE_DIR}/spmv_coo_example.cu
    ${CUSPARSE_DIR}/spmv_csr_example.cu
    #${CUSPARSE_DIR}/spsm_coo_example.cu
    #${CUSPARSE_DIR}/spsm_csr_example.cu
    #${CUSPARSE_DIR}/spsv_coo_example.cu
    #${CUSPARSE_DIR}/spsv_csr_example.cu
    ${CUSPARSE_DIR}/spvv_example.cu

    ${CUSPARSE_TEST_TMP_FILES}
)

