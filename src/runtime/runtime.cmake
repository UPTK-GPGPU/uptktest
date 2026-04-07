set(RUNTIME_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

message(STATUS "include runtime api")

include(${RUNTIME_ROOT_PATH}/device/device.cmake)
include(${RUNTIME_ROOT_PATH}/error/error.cmake)
include(${RUNTIME_ROOT_PATH}/event/event.cmake)
include(${RUNTIME_ROOT_PATH}/graph/graph.cmake)
include(${RUNTIME_ROOT_PATH}/kernel/kernel.cmake)
include(${RUNTIME_ROOT_PATH}/memory/memory.cmake)
include(${RUNTIME_ROOT_PATH}/memory2/memory2.cmake)
include(${RUNTIME_ROOT_PATH}/P2P/P2P.cmake)
include(${RUNTIME_ROOT_PATH}/stream/stream.cmake)

set(RUNTIME_TEST_FILES
    ${RUNTIME_DEVICE_TEST_FILES} 
    ${RUNTIME_ERROR_TEST_FILES}
    ${RUNTIME_EVENT_TEST_FILES}
    ${RUNTIME_GRAPH_TEST_FILES}
    ${RUNTIME_KERNEL_TEST_FILES}
    ${RUNTIME_MEMORY_TEST_FILES}
    ${RUNTIME_MEMORY2_TEST_FILES}
    ${RUNTIME_P2P_TEST_FILES}
    ${RUNTIME_STREAM_TEST_FILES} 

)

#message(STATUS "RUNTIME_TEST_FILES: ${RUNTIME_TEST_FILES}")
