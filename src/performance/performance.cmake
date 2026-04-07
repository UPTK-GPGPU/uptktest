set(PERFORMANCE_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/performance)

message(STATUS "include performance test")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	${PERFORMANCE_ROOT_PATH}/unpin_H2D
	${PERFORMANCE_ROOT_PATH}/unpin_D2H
	${PERFORMANCE_ROOT_PATH}/pin_H2D
	${PERFORMANCE_ROOT_PATH}/pin_D2H
	${PERFORMANCE_ROOT_PATH}/event_performance
	${PERFORMANCE_ROOT_PATH}/hipfree_performance
	${PERFORMANCE_ROOT_PATH}/hipmalloc_performance
	${PERFORMANCE_ROOT_PATH}/hipmemset_performance)


include(unpin_H2D)
include(unpin_D2H)
include(pin_H2D)
include(pin_D2H)
include(event_performance)
include(hipfree_performance)
include(hipmalloc_performance)
include(hipmemset_performance)

set(PERFORMANCE_TEST_FILES
    ${PERFORMANCE_UNPIN_H2D_TEST_FILES}
	${PERFORMANCE_UNPIN_D2H_TEST_FILES}
    ${PERFORMANCE_PIN_H2D_TEST_FILES}
	${PERFORMANCE_PIN_D2H_TEST_FILES}
    ${PERFORMANCE_EVENT_TEST_FILES}
	${PERFORMANCE_HIPFREE_TEST_FILES}
	${PERFORMANCE_HIPMALLOC_TEST_FILES}
	${PERFORMANCE_HIPMEMSET_TEST_FILES})
