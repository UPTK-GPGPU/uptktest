message(STATUS "include extensions test case")
set(EXTENSIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cublas/extensions)

set(EXTENSIONS_TEST_FILES
    ${EXTENSIONS_DIR}/cublas_axpyex_example.cu
    #${EXTENSIONS_DIR}/cublas_cherk3mex_example.cu
    #${EXTENSIONS_DIR}/cublas_cherkex_example.cu
    #${EXTENSIONS_DIR}/cublas_csyrk3mex_example.cu
    #${EXTENSIONS_DIR}/cublas_csyrkex_example.cu
    ${EXTENSIONS_DIR}/cublas_dgmm_example.cu
    ${EXTENSIONS_DIR}/cublas_dotcex_example.cu
    ${EXTENSIONS_DIR}/cublas_dotex_example.cu
    ${EXTENSIONS_DIR}/cublas_geam_example.cu
    ${EXTENSIONS_DIR}/cublas_gemmbatchedex_example.cu
    ${EXTENSIONS_DIR}/cublas_gemmex_example.cu
    ${EXTENSIONS_DIR}/cublas_gemmStridedbatchedex_example.cu
    ${EXTENSIONS_DIR}/cublas_nrm2ex_example.cu
    ${EXTENSIONS_DIR}/cublas_rotex_example.cu
    ${EXTENSIONS_DIR}/cublas_scalex_example.cu
    #${EXTENSIONS_DIR}/cublas_tpttr_example.cu
    #${EXTENSIONS_DIR}/cublas_trttp_example.cu
)

