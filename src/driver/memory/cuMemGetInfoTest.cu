#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuMemory, cuMemGetInfoTest){
    char* A_d = NULL;
    size_t bufferSize = 102400;
//    char* A_Pinned_h;
//    char* A_OSAlloc_h;
    CUresult ret = CUDA_SUCCESS;

    ret = cuMemAlloc((CUdeviceptr*)&A_d, bufferSize);
    EXPECT_EQ(ret, CUDA_SUCCESS);

//    cudaMallocHost((void**)&A_Pinned_h, bufferSize, cudaHostAllocDefault);
//    A_OSAlloc_h = (char*)malloc(bufferSize);

    size_t free, total;
    ret = cuMemGetInfo(&free, &total);//返回设备上的空闲内存和可分配内存总量的快照
    EXPECT_EQ(ret, CUDA_SUCCESS);

    printf("cuMemGetInfo: free=%zu (%4.2f) bufferSize=%lu total=%zu (%4.2f)\n", free,
           (float)(free / 1024.0 / 1024.0), bufferSize, total, (float)(total / 1024.0 / 1024.0));
    EXPECT_TRUE(free + bufferSize <= total);

    cuMemFree((CUdeviceptr)A_d);

}