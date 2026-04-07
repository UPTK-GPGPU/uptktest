message(STATUS "include level-2 test case")

set(CUBLAS_LEVEL_2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cublas/level-2)

set(CUBLAS_LEVEL_2_TEST_FILES
    ${CUBLAS_LEVEL_2_DIR}/cublas_gbmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_gemv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_ger_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_hbmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_hemv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_her_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_her2_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_hpmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_hpr_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_hpr2_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_sbmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_spmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_spr_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_spr2_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_symv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_syr_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_syr2_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_tbmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_tbsv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_tpmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_tpsv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_trmv_example.cu
    ${CUBLAS_LEVEL_2_DIR}/cublas_trsv_example.cu

    ${CUBLAS_LEVEL_2_TEST_TMP_FILES}
)

