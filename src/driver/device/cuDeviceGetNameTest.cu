#include <stdio.h>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define len 256
#define DEVICE_NAME "HIP_DEVICE_NAME"

TEST(cuDevice, cuDeviceGetNameTest)
{
    int numDevices = 0;
    char name[len];
    char deviceName[len] = {"Vega 20"};
    memset(name, 0, len);
    CUdevice device;

    char *device_name = getenv(DEVICE_NAME);
    printf("HIP_DEVICE_NAME :%s\n",device_name);
      
    CUresult ret1 = CUDA_SUCCESS;
    CUresult ret2 = CUDA_SUCCESS;
    cuDeviceGetCount(&numDevices); //numDevices=2
    for (int i = 0; i < numDevices; i++){
        ret1 = cuDeviceGet(&device, i);//device=0和1
        ret2 = cuDeviceGetName(name, len, device);//name=Device 6860,len=256
        #if 0
        if (device_name)
        {
           EXPECT_STREQ(name, device_name);
        }else {
           EXPECT_STREQ(name, deviceName);
        }
        #endif
        printf("device name:%s\n",name);
        EXPECT_EQ(ret1, CUDA_SUCCESS);
        EXPECT_EQ(ret2, CUDA_SUCCESS);

    }
}
