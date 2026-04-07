#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaStream, cudaMultiDevices){

   int numDevices = 0;
   int STREAM_NUM = 0;
   UPTKError_t ret = UPTKSuccess;
   UPTKGetDeviceCount(&numDevices);
   printf("num of devices : %d \n", numDevices);
    //for(int i = 0;;)
   STREAM_NUM = numDevices * 8;
   UPTKStream_t mystream[STREAM_NUM];

   for(int i=0;i<STREAM_NUM;i++){
      ret= UPTKSetDevice(STREAM_NUM%8);
      EXPECT_EQ(ret, UPTKSuccess);
      ret = UPTKStreamCreate(&mystream[i]);
      EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
   }

   for(int i=0;i<STREAM_NUM;i++){
      ret = UPTKStreamDestroy(mystream[i]);
      EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
   }
}
