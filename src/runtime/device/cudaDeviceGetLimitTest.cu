#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
TEST(cudaDevice, cudaDeviceGetLimitTest)
{
    UPTKError_t ret = UPTKSuccess;
    size_t heap;
    ret = UPTKDeviceGetLimit(&heap, UPTKLimitMallocHeapSize);
    EXPECT_EQ(ret, UPTKSuccess);
}