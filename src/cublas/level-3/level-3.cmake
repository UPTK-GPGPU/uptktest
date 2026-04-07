message(STATUS "include level-3 test case")
set(CUBLAS_LEVEL_3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cublas/level-3)

set(CUBLAS_LEVEL_3_TEST_FILES
    ${CUBLAS_LEVEL_3_DIR}/cublas_gemm_example.cu
    #${CUBLAS_LEVEL_3_DIR}/cublas_gemm3m_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_gemmBatched_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_gemmStridedBatched_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_hemm_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_her2k_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_herk_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_herkx_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_symm_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_syr2k_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_syrk_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_syrkx_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_trmm_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_trsm_example.cu
    ${CUBLAS_LEVEL_3_DIR}/cublas_trsmBatched_example.cu
    ${CUBLAS_LEVEL_3_TEST_TMP_FILES}
)

