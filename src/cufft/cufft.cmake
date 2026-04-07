message(STATUS "include cufft test case")
set(CUFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cufft)

set(CUFFT_TEST_FILES
    ${CUFFT_DIR}/1d_c2c_example.cu
    ${CUFFT_DIR}/1d_r2c_example.cu
    ${CUFFT_DIR}/2d_c2r_example.cu
    ${CUFFT_DIR}/3d_c2c_example.cu

    ${CUFFT_TEST_TMP_FILES}
    )

