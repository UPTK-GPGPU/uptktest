#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"


TEST(hipPerformanceEvent,hipEventRecord_Performance){
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    cudaError_t ret = cudaSuccess;
    cudaEvent_t startEvent;

    cudaEventCreate(&startEvent);

    gettimeofday(&start,NULL);
    ret = cudaEventRecord(startEvent, NULL);
    EXPECT_EQ(ret, cudaSuccess);
    gettimeofday(&end,NULL);
    long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    std::cout << "The cost time :" << diff2<<"(us)"<< std::endl;
	
}
