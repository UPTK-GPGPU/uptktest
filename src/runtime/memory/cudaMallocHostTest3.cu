#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMallocHostTest3){
    char *h_A,*h_B;
    char *d_A;
    int bufferSize = 1024;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMallocHost(&h_A,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);

    h_B = (char *)malloc(bufferSize);
    EXPECT_TRUE(h_B!=NULL) << "error: malloc failed";

    ret = UPTKMalloc(&d_A,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);
    
    for(int i=0; i<bufferSize; i++){
	    h_A[i] = 65;
    }

    ret = UPTKMemcpy(d_A, h_A, bufferSize, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy form device to host failed";
    // copy kernel result back to host side
    ret = UPTKMemcpy(h_B, d_A, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy form device to host failed";

    for(int i=0; i<bufferSize; i++){
	    EXPECT_TRUE(h_A[i] == h_B[i]);
    }

    //调用cudaMallocHost 、cudaMalloc申请内存，不释放，退出
}