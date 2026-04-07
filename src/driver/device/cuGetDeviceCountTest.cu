#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuDevice, cuDeviceGetCountTest)
{
    int numDevices = 0;
    CUresult ret = CUDA_SUCCESS;
    ret = cuDeviceGetCount(&numDevices);
    EXPECT_GT(numDevices, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}
