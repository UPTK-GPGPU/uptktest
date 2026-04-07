message(STATUS "include P2P test case")
set(DRIVER_P2P_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/P2P)

set(DRIVER_MODULE_TEST_FILES
    ${DRIVER_P2P_DIR}/cuDeviceCanAccessPeerTest.cu
    ${DRIVER_P2P_DIR}/cuMemGetAddressRangeTest.cu
    ${DRIVER_P2P_TEST_TMP_FILES}
)

