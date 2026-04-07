message(STATUS "include device_access_memory test case")
set(RUNTIME_P2P_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/P2P)

set(RUNTIME_P2P_TEST_FILES
      ${RUNTIME_P2P_DIR}/cudaDeviceCanAccessPeerTest.cu
      ${RUNTIME_P2P_DIR}/cudaDeviceDisablePeerAccessTest.cu
      ${RUNTIME_P2P_DIR}/cudaMemcpyPeerAsyncTest.cu
      ${RUNTIME_P2P_DIR}/cudaMemcpyPeerAsyncTest2.cu
      ${RUNTIME_P2P_DIR}/cudaMemcpyPeerAsyncTest3.cu
      ${RUNTIME_P2P_DIR}/cudaMemcpyPeerTest.cu
  )
