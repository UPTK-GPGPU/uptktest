#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaErrorDevice, cudaErrorGetErrorNameTest)
{
    UPTKError_t ret1 = UPTKSuccess;
    
    int deviceId = -1;
    ret1 = UPTKSetDevice(deviceId);
    EXPECT_EQ(ret1, UPTKErrorInvalidDevice);
    std::cout<<ret1;
    std::cout<<UPTKGetErrorName(ret1);
    
    //UPTKDeviceProp props;
    //ret = UPTKGetDeviceProperties(&props, deviceId));
    //printf("info: running on device #%d %s\n", deviceId, props.name);
}
