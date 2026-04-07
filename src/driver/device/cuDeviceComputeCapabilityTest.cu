#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuDevice, cuDeviceComputeCapabilityTest)
{
    int numDevices = 0;
    int major,minor;
    CUdevice device;
    CUresult ret1 = CUDA_SUCCESS;
    CUresult ret2 = CUDA_SUCCESS;
    cuDeviceGetCount(&numDevices); //numDevices=2
    for (int i = 0; i < numDevices; i++){
        ret1 = cuDeviceGet(&device, i);
        ret2 = cuDeviceComputeCapability(&major, &minor, device);//major=9,minor=0
        //std::cout << "major=" << major << std::endl;
        //std::cout << "minor=" << minor << std::endl;
        EXPECT_TRUE(major >= 0);
        EXPECT_TRUE(minor >= 0);
        EXPECT_EQ(ret1, CUDA_SUCCESS);
        EXPECT_EQ(ret2, CUDA_SUCCESS);

    }
}
