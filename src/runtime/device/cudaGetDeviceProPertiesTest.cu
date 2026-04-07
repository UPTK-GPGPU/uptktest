#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaGetDevicePropertiesTest) {
    UPTKDeviceProp props;
    int deviceID = 1;
    UPTKError_t ret = UPTKSuccess;
    UPTKGetDeviceProperties(&props, deviceID);
    EXPECT_EQ(ret, UPTKSuccess);
    
}