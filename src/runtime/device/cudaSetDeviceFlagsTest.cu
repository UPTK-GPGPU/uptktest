#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaSetDeviceFlagsTest)
{
    unsigned flag = 0;
    UPTKError_t ret = UPTKSuccess;
    UPTKDeviceReset();
    int deviceCount = 0;
    UPTKGetDeviceCount(&deviceCount);
    for(int j = 0; j < deviceCount; j++){
        UPTKSetDevice(j);
        for(int i = 0; i < 4; i++){
            flag = 1 << i;
            //printf("Flag=%x\n", flag);
            UPTKSetDeviceFlags(flag);
            EXPECT_EQ(ret, UPTKSuccess);
        }
        flag = 0;
    }
}
