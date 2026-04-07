message(STATUS "include level-1 test case")
set(CUBLAS_LEVEL_1_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cublas/level-1)

set(CUBLAS_LEVEL_1_TEST_FILES
    ${CUBLAS_LEVEL_1_DIR}/cublas_amax_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_amin_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_asum_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_axpy_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_copy_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_dot_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_dotc_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_nrm2_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_rot_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_rotg_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_rotm_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_rotmg_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_scal_example.cu
    ${CUBLAS_LEVEL_1_DIR}/cublas_swap_example.cu
    ${CUBLAS_LEVEL_1_TEST_TMP_FILES}
)

