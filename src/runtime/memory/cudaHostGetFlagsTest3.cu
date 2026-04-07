#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define LEN 100

TEST(cudaMemory,cudaHostGetFlagsTest3){
    float *h_A = NULL;
    float *d_A = NULL;
    unsigned int flagA, FlagA;
    FlagA = UPTKHostAllocWriteCombined | UPTKHostAllocMapped;
    int bufferSize = LEN*sizeof(float);
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMallocHost((void**)&h_A, bufferSize, UPTKHostAllocWriteCombined | UPTKHostAllocMapped);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocHost failed";

    // initialize data at host side
    for (int i = 0; i < LEN; i++)
    {
        h_A[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    ret = UPTKHostGetFlags(&flagA, h_A);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostGetFlags failed";
    EXPECT_EQ(flagA, FlagA);
    //EXPECT_EQ(flagA, flagA);

    ret = UPTKFree(h_A);
    EXPECT_EQ(ret, UPTKSuccess)  << "call UPTKFree failed";
}
