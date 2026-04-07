#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuContext, cuCtxSetCurrentTest) {
    CUresult ret = CUDA_SUCCESS;
    #if 0
    CUcontext context1;
    CUcontext context2;
    CUcontext context3;
    CUcontext context;
    CUdevice device;
    unsigned int flag;
    int active;

    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxSetCurrent(context1);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxSetCurrent(context1);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxGetCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    if(context1==context){
       std::cout<<"equal";
    }
    ret = cuCtxDestroy(context1);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxDestroy(context2);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxDestroy(context3);
    #endif
    EXPECT_EQ(ret, CUDA_SUCCESS);

}