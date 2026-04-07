#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuContext, cuDevicePrimaryCtxReleaseTest) {
    CUresult ret = CUDA_SUCCESS;
    CUcontext context;
    CUdevice device;
    unsigned int flag;
    int active;
    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxCreate(&context, 0, device);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    
    ret = cuDevicePrimaryCtxRelease(device);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    
    //Warning：This function return #CUDA_SUCCESS though doesn’t release the primaryCtx by design on HIP/HCC path.
    ret = cuDevicePrimaryCtxGetState(device,&flag,&active);
    EXPECT_EQ(ret, CUDA_SUCCESS);    
    std::cout<<flag;     //Although Ctx is released, flag can still be obtained because of the presence of warnings

    ret = cuCtxDestroy(context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
}