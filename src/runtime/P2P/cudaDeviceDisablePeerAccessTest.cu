#include <stdio.h>
#include <iostream>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaP2P,cudaDeviceDisablePeerAccessTest)
{
    UPTKError_t ret = UPTKSuccess;
    #ifdef __TEST_HIPHSA__
    int deviceCnt;
    UPTKGetDeviceCount(&deviceCnt);
    int canAccessPeer;
    unsigned flag = 0;
   
    for (int i = 0; i < deviceCnt; i++) {
      for(int j = i+1; j < deviceCnt; j++){
        int canAccessPeer;
        ret = UPTKDeviceCanAccessPeer(&canAccessPeer, i, j);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKDeviceEnablePeerAccess(j,flag);
        EXPECT_EQ(ret, UPTKSuccess);

        ret = UPTKDeviceDisablePeerAccess(j);
        EXPECT_EQ(ret, UPTKSuccess);
        //ret = UPTKDeviceCanAccessPeer(&canAccessPeer, i, j);
        //EXPECT_EQ(ret, UPTKSuccess);
      }
    }
    #else
    EXPECT_EQ(ret, UPTKSuccess);
    #endif
}