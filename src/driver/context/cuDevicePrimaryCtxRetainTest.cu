#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuContext, cuDevicePrimaryCtxRetainTest) {
    CUresult ret = CUDA_SUCCESS;
    CUcontext context;
    CUdevice device;
    unsigned int flag;
    int active;
    CUcontext  pctx;
    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxCreate(&context, 0, device);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    
    ret = cuDevicePrimaryCtxRelease(device);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    
    ret = cuDevicePrimaryCtxRetain(&pctx,device);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxDestroy(context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}