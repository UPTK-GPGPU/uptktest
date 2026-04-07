message(STATUS "include event test case")
set(DRIVER_EVENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/event)

set(DRIVER_EVENT_TEST_FILES
    ${DRIVER_EVENT_DIR}/cuEventCreateWithFlagsTest.cu
    ${DRIVER_EVENT_TEST_TMP_FILES}
)

