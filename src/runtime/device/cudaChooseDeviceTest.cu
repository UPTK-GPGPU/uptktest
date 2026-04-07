#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
namespace{
TEST(cudaDevice, cudaChooseDeviceTest) {
    UPTKDeviceProp prop;
    int dev;
    UPTKError_t ret = UPTKSuccess;
    UPTKGetDevice(&dev);
    //printf("ID of current HIP device:  %d\n", dev);

    memset(&prop, 0, sizeof(UPTKDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    ret = UPTKChooseDevice(&dev, &prop);
    EXPECT_EQ(ret, UPTKSuccess);
    //printf("ID of hip device closest to revision 1.3:  %d\n", dev);

    ret = UPTKSetDevice(dev);
    EXPECT_EQ(ret, UPTKSuccess);
    
}
}
