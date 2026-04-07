message(STATUS "include event test case")
set(RUNTIME_EVENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/event)

set(RUNTIME_EVENT_TEST_FILES
    ${RUNTIME_EVENT_DIR}/cudaEventCreateTest.cu
    ${RUNTIME_EVENT_DIR}/cudaEventCreateWithFlagsTest.cu
    ${RUNTIME_EVENT_DIR}/cudaEventElapsedTimeTest.cu
    ${RUNTIME_EVENT_DIR}/cudaEventPassBigST.cu
    ${RUNTIME_EVENT_DIR}/cudaEventQueryTest.cu
    ${RUNTIME_EVENT_DIR}/cudaEventRecordTest.cu
    ${RUNTIME_EVENT_DIR}/cudaEventSynchronizeTest.cu
    ${RUNTIME_EVENT_TEST_TMP_FILES}
)
      
