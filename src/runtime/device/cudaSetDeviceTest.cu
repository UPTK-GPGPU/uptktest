#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaDevice, cudaSetDeviceTest)
{
    int numDevices = 0;
    UPTKError_t ret = UPTKSuccess;
    UPTKGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
       ret= UPTKSetDevice(i);
       EXPECT_EQ(ret, UPTKSuccess);
       ret = UPTKGetDevice(&i);
       EXPECT_EQ(ret, UPTKSuccess);
    }
    ret = UPTKSetDevice(numDevices);
    EXPECT_EQ(ret, UPTKErrorInvalidDevice);
}
