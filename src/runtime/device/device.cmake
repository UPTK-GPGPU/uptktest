message(STATUS "include device test case")
set(RUNTIME_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/device)

set(RUNTIME_DEVICE_TEST_FILES
      ${RUNTIME_DEVICE_DIR}/cudaChooseDeviceTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceGetAttributeTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceGetByPCIBusIdTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceGetLimitTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceGetPCIBusIdTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceGetSharedMemConfigTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaDeviceResetTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaGetDeviceCountTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaGetDeviceProPertiesTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaGetDeviceTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaRuntimeGetVersionTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaSetDeviceFlagsTest.cu
      ${RUNTIME_DEVICE_DIR}/cudaSetDeviceTest.cu
      ${RUNTIME_DEVICE_TEST_TMP_FILES} 
  )
