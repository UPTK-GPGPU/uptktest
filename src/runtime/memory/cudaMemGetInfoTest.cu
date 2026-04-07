#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory, cudaMemGetInfoTest){
    char* A_d = NULL;
    size_t bufferSize = 102400;
//    char* A_Pinned_h;
//    char* A_OSAlloc_h;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMalloc(&A_d, bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);

//    UPTKMallocHost((void**)&A_Pinned_h, bufferSize, UPTKHostAllocDefault);
//    A_OSAlloc_h = (char*)malloc(bufferSize);

    size_t free, total;
    ret = UPTKMemGetInfo(&free, &total);//返回设备上的空闲内存和可分配内存总量的快照
    EXPECT_EQ(ret, UPTKSuccess);

    printf("UPTKMemGetInfo: free=%zu (%4.2f) bufferSize=%lu total=%zu (%4.2f)\n", free,
           (float)(free / 1024.0 / 1024.0), bufferSize, total, (float)(total / 1024.0 / 1024.0));
    EXPECT_TRUE(free + bufferSize <= total);

    UPTKFree(A_d);

}

