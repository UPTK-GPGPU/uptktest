#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuDevice, cuDeviceGetTest)
{
    int numDevices = 0;
    CUdevice device;
    CUresult ret1 = CUDA_SUCCESS;
    CUresult ret2 = CUDA_SUCCESS;
    cuDeviceGetCount(&numDevices); //numDevices=2
    for (int i = 0; i < numDevices; i++){
        ret1 = cuDeviceGet(&device, i);
        EXPECT_EQ(ret1, CUDA_SUCCESS);
        EXPECT_EQ(device,i);
    }
}