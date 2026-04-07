#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime_api.h"


TEST(hipPerformancehipMemset,hipPerformancehipMemset_1){
#if 1 
    int numDevice = 0;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    char *tempbuffer = NULL;
    char *tempbuffer2 = NULL;
    int bufferSize = 1;//2的30次方

    cudaError_t ret = cudaSuccess;
    int device =0;
    
    ret =cudaGetDevice(&device);
    EXPECT_EQ(ret, cudaSuccess);
   
    ret = cudaMalloc(&tempbuffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    
    ret = cudaMalloc(&tempbuffer2, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaMemset(tempbuffer,0,bufferSize);
    gettimeofday(&start,NULL);
    ret = cudaMemset(tempbuffer2,0,bufferSize);
    gettimeofday(&end,NULL);
    EXPECT_EQ(ret, cudaSuccess);	
    long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;

    //gettimeofday(&start,NULL);
    ret = cudaFree(tempbuffer);
    EXPECT_EQ(ret, cudaSuccess);	
    //gettimeofday(&end,NULL);
    //long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;

    ret = cudaFree(tempbuffer2);
    EXPECT_EQ(ret, cudaSuccess);	
    //long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<< std::endl;
#else

    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    int num = 1;

    char *tempbuffer = NULL;

    int bufferSize = 1024*1024;//2的30次方
    //int bufferSize = 2147483648;//2的31次方
    cudaError_t ret = cudaSuccess;

    //ret = cudaMallocHost(&hostBuffer, bufferSize);
    //EXPECT_EQ(ret, cudaSuccess);
    //ASSERT_TRUE(hostBuffer!=NULL);

    //for(int i=0; i<bufferSize; i++)
    //{
    //	    hostBuffer[i]   = i;
    // }

    //tempbuffer = (char **)malloc(sizeof(char *) * num);
    //for (int i = 0; i<num; i++)
    {
       ret = cudaMalloc(&tempbuffer, bufferSize);
       EXPECT_EQ(ret, cudaSuccess);
    }

    //cudaPointerAttributes attribs;
    //memset(&attribs, 0, sizeof(cudaPointerAttributes));

    //ret = cudaPointerGetAttributes(&attribs,(void *)tempbuffer[i]);
    //EXPECT_EQ(ret, cudaSuccess);
    //EXPECT_EQ((char *)attribs.devicePointer,buffer);

    //std::cout << "buffer=" << (void *)buffer << std::endl;
    for(int i=0; i< num;i++)
    {
        //ret = cudaMemcpy(tempbuffer[i], hostBuffer, bufferSize, cudaMemcpyHostToDevice);	    
        //gettimeofday(&start,NULL);
        //ret = cudaMemcpy(tempbuffer[i], hostBuffer, bufferSize, cudaMemcpyHostToDevice);
        //EXPECT_EQ(ret, cudaSuccess);
        //gettimeofday(&end,NULL);
        //long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
        //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<<",index: "<<i<< std::endl;
        //diff += diff2;
    }
   // std::cout << "The cost time :" << (double)((double)diff/(double)num)<<"(us), size: "<<bufferSize<< std::endl;
    gettimeofday(&start,NULL);
    //for (int i = 0; i<num; i++)
    {
        ret = cudaFree(tempbuffer);
        EXPECT_EQ(ret, cudaSuccess);	
    }
    gettimeofday(&end,NULL);
    long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
    std::cout << "The cost time :" << (double)((double)diff/(double)num)<<"(us), size: "<<bufferSize<< std::endl;
    //cudaFree(hostBuffer);
    //free(tempbuffer);
#endif
}
