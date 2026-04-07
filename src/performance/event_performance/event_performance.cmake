message(STATUS "include unpin_H2D test case")
set(PERFORMANCE_EVENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/performance/event_performance)

set(PERFORMANCE_EVENT_TEST_FILES 
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_2.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_3.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_4.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_5.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_6.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_7.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_8.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_9.cpp
    ${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_10.cpp
    ${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_11.cpp
    ${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_12.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_13.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_14.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_15.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_16.cpp
	${PERFORMANCE_EVENT_DIR}/hipEventRecord_performance_17.cpp
	)
