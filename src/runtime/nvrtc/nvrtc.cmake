message(STATUS "include nvrtc test case")
set(RUNTIME_NVRTC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/nvrtc)

set(RUNTIME_NVRTC_TEST_FILES
    ${RUNTIME_NVRTC_DIR}/warpsizeTest.cu
    # ${RUNTIME_NVRTC_DIR}/customOptionsTest.cu
    ${RUNTIME_NVRTC_DIR}/nvRtcFunctionalTest.cu
    ${RUNTIME_NVRTC_DIR}/saxpyTest.cu
    ${RUNTIME_NVRTC_DIR}/includepathTest.cu
    ${RUNTIME_RTC_TEST_TMP_FILES}
)

# copy file for includepathTest.cu
file(COPY ${RUNTIME_NVRTC_DIR}/saxpy.h DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${RUNTIME_NVRTC_DIR}/headers DESTINATION ${CMAKE_BINARY_DIR})