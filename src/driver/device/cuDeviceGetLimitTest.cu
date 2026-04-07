#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
TEST(cuDevice, cuDeviceGetLimitTest)
{
    CUresult ret = CUDA_SUCCESS;
    size_t heap;
    ret = cuCtxGetLimit(&heap, CU_LIMIT_MALLOC_HEAP_SIZE);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}