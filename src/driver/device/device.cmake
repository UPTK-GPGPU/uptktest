message(STATUS "include device test case")
set(DRIVER_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/device)

set(DRIVER_DEVICE_TEST_FILES
    ${DRIVER_DEVICE_DIR}/cuDeviceComputeCapabilityTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceGetByPCIBusIdTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceGetLimitTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceGetNameTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceGetPCIBusIdTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceGetTest.cu
    ${DRIVER_DEVICE_DIR}/cuDeviceTotalMemTest.cu
    ${DRIVER_DEVICE_DIR}/cuDriverGetVersionTest.cu
    ${DRIVER_DEVICE_DIR}/cuGetDeviceCountTest.cu
    ${DRIVER_DEVICE_TEST_TMP_FILES}
)

