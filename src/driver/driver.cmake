set(DRIVER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/driver)

message(STATUS "include driver api")

include(${DRIVER_ROOT_PATH}/context/context.cmake)
include(${DRIVER_ROOT_PATH}/device/device.cmake)
include(${DRIVER_ROOT_PATH}/event/event.cmake)
include(${DRIVER_ROOT_PATH}/memory/memory.cmake)
include(${DRIVER_ROOT_PATH}/module/module.cmake)
include(${DRIVER_ROOT_PATH}/P2P/P2P.cmake)

set(DRIVER_TEST_FILES 
    ${DRIVER_CONTEXT_TEST_FILES}
    ${DRIVER_DEVICE_TEST_FILES} 
    ${DRIVER_EVENT_TEST_FILES}
    ${DRIVER_MEMORY_TEST_FILES}
    ${DRIVER_MODULE_TEST_FILES}
    ${DRIVER_P2P_TEST_FILES}
)

#message(STATUS "DRIVER_TEST_FILES: ${DRIVER_TEST_FILES}")
