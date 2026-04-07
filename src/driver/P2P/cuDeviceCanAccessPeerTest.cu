#include <stdio.h>
#include <iostream>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuP2P,cuDeviceCanAccessPeerTest)
{
    CUresult ret = CUDA_SUCCESS;
    int deviceCnt;
    cuDeviceGetCount(&deviceCnt);
    int canAccessPeer;

    
    for (int i = 0; i < deviceCnt; i++) {
       for(int j = i+1; j < deviceCnt; j++){
        int canAccessPeer;
        cuDeviceCanAccessPeer(&canAccessPeer, (CUdevice)i, (CUdevice)j);
        EXPECT_EQ(ret, CUDA_SUCCESS);
         //if(ret==cudaSuccess){
         //std::cout<<"device#" << i << " ,"<<j<<"can access.\n";
        //}else{
        //std::cout << "device#" << i << " ,"<<j<<"can not access.\n";
       //}
       
      }
    }

}