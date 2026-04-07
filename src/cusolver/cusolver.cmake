message(STATUS "include cusolver test case")
set(CUSOLVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cusolver)

set(CUSOLVER_TEST_FILES
    ${CUSOLVER_DIR}/cusolver_gesvd_example.cu
    ${CUSOLVER_DIR}/cusolver_getrf_example.cu
    ${CUSOLVER_DIR}/cusolver_orgqr_example.cu
    ${CUSOLVER_DIR}/cusolver_ormqr_example.cu
    ${CUSOLVER_DIR}/cusolver_potrfBatched_example.cu
    ${CUSOLVER_DIR}/cusolver_syevd_example.cu
    ${CUSOLVER_DIR}/cusolver_sygvd_example.cu
    ${CUSOLVER_TEST_TMP_FILES}
)

