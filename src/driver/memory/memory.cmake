message(STATUS "include memory test case")
set(DRIVER_MEMORY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/memory)

set(DRIVER_MEMORY_TEST_FILES
    ${DRIVER_MEMORY_DIR}/cuMemcpy2DTest3.cu
    ${DRIVER_MEMORY_DIR}/cuMemGetInfoTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoDAsyncTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoDAsyncTest2.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoDTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoHAsyncTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoHAsyncTest1.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyDtoHTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyHtoDAsyncTest.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyHtoDAsyncTest1.cu
    ${DRIVER_MEMORY_DIR}/cuMemcpyHtoDTest.cu
    ${DRIVER_MEMORY_TEST_TMP_FILES}
)