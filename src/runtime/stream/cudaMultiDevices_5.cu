#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define STREAM_NUM 4

TEST(cudaStream, cudaMultiDevices_5){

    UPTKError_t ret = UPTKSuccess;

    int numDevices = 0;
    UPTKGetDeviceCount(&numDevices);
    if(numDevices!=4){
        printf("Pass \n");
        return;
    }

    UPTKStream_t mystream[STREAM_NUM];
    for(int i=0;i<numDevices;i++){
        ret= UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKStreamCreate(&mystream[i]);
        EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }

    for(int i=0;i<numDevices;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }




    
}
