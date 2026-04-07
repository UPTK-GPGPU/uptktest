#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(UPTKError_t, cudaGetLastErrorTest)
{
    UPTKError_t ret1 = UPTKSuccess;
    UPTKError_t ret2 = UPTKSuccess;
    UPTKError_t ret3 = UPTKSuccess;
    int deviceId1 = 4;
    int deviceId2 = -1;
    
    ret1 = UPTKGetDevice(&deviceId1);
    EXPECT_NE(ret1, UPTKErrorInvalidDevice);
    ret2 = UPTKSetDevice(deviceId2);
    EXPECT_EQ(ret2, UPTKErrorInvalidDevice);
    ret3 = UPTKGetLastError();
    EXPECT_EQ(ret3, UPTKErrorInvalidDevice);
    
}