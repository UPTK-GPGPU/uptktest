#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"


TEST(hipPerformancehipfree,hipPerformancehipfree_13){
#if 0
    int numDevice = 0;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    char *tempbuffer = NULL;
    int bufferSize = 1024*4;//2的30次方
    cudaError_t ret = cudaSuccess;
    int device =0;
    
    ret =cudaGetDevice(&device);
    EXPECT_EQ(ret, cudaSuccess);
   
    ret = cudaMalloc(&tempbuffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    
    gettimeofday(&start,NULL);
    ret = cudaFree(tempbuffer);
    EXPECT_EQ(ret, cudaSuccess);	
    gettimeofday(&end,NULL);
    long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;
    #else
    int numDevice = 0;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    char *tempbuffer = NULL;
    char *tempbuffer2 = NULL;
    int bufferSize = 1024*4;//2的30次方
    cudaError_t ret = cudaSuccess;
    int device =0;
    
    ret =cudaGetDevice(&device);
    EXPECT_EQ(ret, cudaSuccess);
   
    ret = cudaMalloc(&tempbuffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    
    ret = cudaMalloc(&tempbuffer2, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);

    //gettimeofday(&start,NULL);
    ret = cudaFree(tempbuffer);
    EXPECT_EQ(ret, cudaSuccess);	
    //gettimeofday(&end,NULL);
    //long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;

    gettimeofday(&start,NULL);
    ret = cudaFree(tempbuffer2);
    EXPECT_EQ(ret, cudaSuccess);	
    gettimeofday(&end,NULL);
    long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;
    #endif
}