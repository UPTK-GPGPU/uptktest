#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(UPTKError_t, cudaGetErrorStringTest)
{
    UPTKError_t ret = UPTKSuccess;
    //UPTKError_t ret2 = UPTKSuccess;
    int deviceId = -1;
    ret = UPTKSetDevice(deviceId);
    EXPECT_EQ(ret, UPTKErrorInvalidDevice);
    std::cout<<UPTKGetErrorString(ret);
    //std::cout<<ret2;
    //UPTKDeviceProp props;
    //ret = UPTKGetDeviceProperties(&props, deviceId));
    //printf("info: running on device #%d %s\n", deviceId, props.name);
}
