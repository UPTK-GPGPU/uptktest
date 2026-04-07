message(STATUS "include context test case")
set(DRIVER_CONTEXT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/context)

set(DRIVER_CONTEXT_TEST_FILES
    ${DRIVER_CONTEXT_DIR}/cuCtxCreateTest.cu
    # BUG # ${DRIVER_CONTEXT_DIR}/cuCtxPushCurrentTest.cu
    ${DRIVER_CONTEXT_DIR}/cuCtxSetCurrentTest.cu
    ${DRIVER_CONTEXT_DIR}/cuDevicePrimaryCtxGetStateTest.cu
    ${DRIVER_CONTEXT_DIR}/cuDevicePrimaryCtxReleaseTest.cu
    ${DRIVER_CONTEXT_DIR}/cuDevicePrimaryCtxResetTest.cu
    ${DRIVER_CONTEXT_DIR}/cuDevicePrimaryCtxRetainTest.cu
    # hip TODO # ${DRIVER_CONTEXT_DIR}/cuDevicePrimaryCtxSetFlagsTest.cu
    ${DRIVER_CONTEXT_TEST_TMP_FILES}
)

