#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuContext, cuDevicePrimaryCtxResetTest) {
    CUresult ret = CUDA_SUCCESS;
    CUcontext context;
    CUdevice device;
    unsigned int flag;
    int active;
    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxCreate(&context, 0, device);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuDevicePrimaryCtxReset(device);
    EXPECT_EQ(ret, CUDA_SUCCESS);
   
    ret = cuDevicePrimaryCtxGetState(device,&flag,&active);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    std::cout<<flag;
    std::cout<<active;

    ret = cuCtxDestroy(context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}