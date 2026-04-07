#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace {
TEST(cuContext, cuCtxCreateTest) {
    CUresult ret = CUDA_SUCCESS;
    CUcontext context;
    CUdevice device;
    int flag;
    int active;
    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxCreate(&context, 0, device);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxDestroy(context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}
}
