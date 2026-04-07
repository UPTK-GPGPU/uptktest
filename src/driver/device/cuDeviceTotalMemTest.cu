#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuDevice,cuDeviceTotalMemTest){
    int numDevices = 0;
    size_t totMem;
    CUresult ret = CUDA_SUCCESS;
    CUdevice device;
    cuDeviceGetCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
        cuDeviceGet(&device, i);
        ret = cuDeviceTotalMem(&totMem, device);//返回设备上的内存总量，totMem=17163091968,totMem=17163091968
        //std::cout << "totMem=" << totMem << std::endl;
        EXPECT_EQ(ret, CUDA_SUCCESS);
        EXPECT_TRUE(totMem != 0);
    }
}