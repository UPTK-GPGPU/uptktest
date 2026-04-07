#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"


TEST(hipPerformanceUnpinH2D,hipPerformanceUnpinH2D_15){
    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    int num = 1;

    int bufferSize = 1024*16;//2的30次方
    //int bufferSize = 2147483648;//2的31次方
    cudaError_t ret = cudaSuccess;

    hostBuffer = (char *)malloc(bufferSize);
    ASSERT_TRUE(hostBuffer!=NULL);

    for(int i=0; i<bufferSize; i++)
    {
	    hostBuffer[i]   = i;
    }

    ret = cudaMalloc(&buffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);

    cudaPointerAttributes attribs;
    memset(&attribs, 0, sizeof(cudaPointerAttributes));

    ret = cudaPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ((char *)attribs.devicePointer,buffer);
    ret = cudaMemcpy(buffer, hostBuffer, bufferSize, cudaMemcpyHostToDevice);
    EXPECT_EQ(ret, cudaSuccess);
    //std::cout << "buffer=" << (void *)buffer << std::endl;
    for(int i=0; i< num;i++)
    {	    
        gettimeofday(&start,NULL);
        ret = cudaMemcpy(buffer, hostBuffer, bufferSize, cudaMemcpyHostToDevice);
        EXPECT_EQ(ret, cudaSuccess);
        gettimeofday(&end,NULL);
        diff += ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
	
    }
    std::cout << "The cost time :" << (double)((double)diff/(double)num)<<"(us), size: "<<bufferSize<< std::endl;
    free(hostBuffer);
    
    ret = cudaFree(buffer);
    EXPECT_EQ(ret, cudaSuccess);	
}
