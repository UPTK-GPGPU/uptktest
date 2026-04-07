#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"


TEST(hipPerformancepinH2D,hipPerformancepinH2D_1){
    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    int num = 1;

    char **tempbuffer = NULL;

    int bufferSize = 1;//2的30次方
    //int bufferSize = 2147483648;//2的31次方
    cudaError_t ret = cudaSuccess;

    ret = cudaMallocHost(&hostBuffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    ASSERT_TRUE(hostBuffer!=NULL);

    for(int i=0; i<bufferSize; i++)
    {
	    hostBuffer[i]   = i;
    }

    tempbuffer = (char **)malloc(sizeof(char *) * num);
    for (int i = 0; i<num; i++)
    {
       ret = cudaMalloc(&tempbuffer[i], bufferSize);
       EXPECT_EQ(ret, cudaSuccess);
    }

    //cudaPointerAttributes attribs;
    //memset(&attribs, 0, sizeof(cudaPointerAttributes));

    //ret = cudaPointerGetAttributes(&attribs,(void *)tempbuffer[i]);
    //EXPECT_EQ(ret, cudaSuccess);
    //EXPECT_EQ((char *)attribs.devicePointer,buffer);
    ret = cudaMemcpy(tempbuffer[0], hostBuffer, bufferSize, cudaMemcpyHostToDevice);
    EXPECT_EQ(ret, cudaSuccess);
    //std::cout << "buffer=" << (void *)buffer << std::endl;
    for(int i=0; i< num;i++)
    {	    
        gettimeofday(&start,NULL);
        ret = cudaMemcpy(tempbuffer[i], hostBuffer, bufferSize, cudaMemcpyHostToDevice);
        EXPECT_EQ(ret, cudaSuccess);
        gettimeofday(&end,NULL);
        long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
        //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<<",index: "<<i<< std::endl;
        diff += diff2;
    }
    std::cout << "The cost time :" << (double)((double)diff/(double)num)<<"(us), size: "<<bufferSize<< std::endl;
    cudaFree(hostBuffer);
    
    for (int i = 0; i<num; i++)
    {
        ret = cudaFree(tempbuffer[i]);
        EXPECT_EQ(ret, cudaSuccess);	
    }
    free(tempbuffer);
}
