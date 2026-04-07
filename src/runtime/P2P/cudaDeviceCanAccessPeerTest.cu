#include <stdio.h>
#include <iostream>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaP2P,cudaDeviceCanAccessPeerTest)
{
    UPTKError_t ret = UPTKSuccess;
    int deviceCnt;
    UPTKGetDeviceCount(&deviceCnt);
    int canAccessPeer;

    
    for (int i = 0; i < deviceCnt; i++) {
       for(int j = i+1; j < deviceCnt; j++){
        int canAccessPeer;
        UPTKDeviceCanAccessPeer(&canAccessPeer, i, j);
        EXPECT_EQ(ret, UPTKSuccess);
         //if(ret==UPTKSuccess){
         //std::cout<<"device#" << i << " ,"<<j<<"can access.\n";
        //}else{
        //std::cout << "device#" << i << " ,"<<j<<"can not access.\n";
       //}
       
      }
    }

}