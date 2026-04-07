#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuContext, cuCtxPushCurrentTest) {
    CUresult ret = CUDA_SUCCESS;
    CUcontext context1;
    CUcontext context2;
    CUcontext context3;
    CUcontext context;
    CUdevice device;
    unsigned int flag;
    int active;
    
    ret = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxPushCurrent(context1);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxGetCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    if(context1==context){
      std::cout<<"equal";
    }

    ret = cuCtxPushCurrent(context2);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxGetCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    if(context2==context){
      std::cout<<"equal";
    }

    ret = cuCtxPushCurrent(context3);
    ret = cuCtxGetCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    if(context3==context){
      std::cout<<"equal";
    }
    
    ret = cuCtxPopCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxGetCurrent(&context);
    EXPECT_EQ(ret, CUDA_SUCCESS);

    ret = cuCtxDestroy(context1);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxDestroy(context2);
    EXPECT_EQ(ret, CUDA_SUCCESS);
    ret = cuCtxDestroy(context3);
    EXPECT_EQ(ret, CUDA_SUCCESS);

}