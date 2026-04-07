message(STATUS "include module test case")
set(DRIVER_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/module)

set(DRIVER_MODULE_TEST_FILES
    ${DRIVER_MODULE_DIR}/cuModuleLoadDataExTest.cu
    ${DRIVER_MODULE_DIR}/cuModuleLoadDataTest.cu
    ${DRIVER_MODULE_DIR}/cuModuleLoadTest.cu
    ${DRIVER_MODULE_TEST_TMP_FILES}
)

