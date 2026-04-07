#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaGetDeviceCountTest)
{
    int numDevices = 0;
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKGetDeviceCount(&numDevices);
    EXPECT_GT(numDevices, 0);
    EXPECT_EQ(ret, UPTKSuccess);
}
