#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaGetDeviceTest)
{
    int deviceID = 0;
    UPTKError_t ret = UPTKSuccess;
    ret= UPTKGetDevice(&deviceID);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ(deviceID, 0);
}

