message(STATUS "include stream test case")
set(RUNTIME_STREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime/stream)

set(RUNTIME_STREAM_TEST_FILES
	${RUNTIME_STREAM_DIR}/cudaMultiDevices.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_1.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_2.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_3.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_4.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_5.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_6.cu
	${RUNTIME_STREAM_DIR}/cudaMultiDevices_7.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_0.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_1.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_2.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_3.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_4.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_5.cu
	${RUNTIME_STREAM_DIR}/cudaMultiStreamTest_6.cu
	${RUNTIME_STREAM_DIR}/cudaMultithreadsTest.cu
	${RUNTIME_STREAM_DIR}/cudaMultithreadsTest_1.cu
	${RUNTIME_STREAM_DIR}/cudaMultithreadsTest_2.cu
	${RUNTIME_STREAM_DIR}/cudaMultithreadsTest_3.cu
	${RUNTIME_STREAM_DIR}/cudaStreamAddCallbackTest.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest10.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest11.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest12.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest2.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest3.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest4.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest5.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest6.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest7.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest8.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateTest9.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateWithFlagsTest.cu
	${RUNTIME_STREAM_DIR}/cudaStreamCreateWithPriorityTest.cu
	${RUNTIME_STREAM_DIR}/cudaStreamGetFlagsTest.cu
	${RUNTIME_STREAM_DIR}/cudaStreamQueryTest.cu
	#${RUNTIME_STREAM_DIR}/cudaStreamWaitEventTest.cu
	)



