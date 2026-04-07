#include <stdio.h>
#include <iostream>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
#define N 32



TEST(cudaP2P,cudaMemcpyPeerTest)
{
    UPTKError_t ret = UPTKSuccess;
    #ifdef __TEST_HIPHSA__
    size_t Nbytes = N * sizeof(int);
    int numDevices = 0;
    int *A_d, *X_d;
    int *A_h, *C_h;
  
    ret = UPTKGetDeviceCount(&numDevices);
    std::cout<<"Device count : "<<numDevices<<"\n";
    if (numDevices <= 1)
    {
       EXPECT_EQ(ret, UPTKSuccess);
       return ;
    }
    
        UPTKMallocHost(&A_h, Nbytes);
        UPTKMallocHost(&C_h, Nbytes);

        UPTKSetDevice(0);
        UPTKMalloc(&A_d, Nbytes);
        UPTKSetDevice(1);
        UPTKMalloc(&X_d, Nbytes);

        for(int i = 0;i<N;i++){
             A_h[i] = 1;
        }
     
       ret = UPTKMemcpy(A_d, A_h, Nbytes, UPTKMemcpyHostToDevice);
       EXPECT_EQ(ret, UPTKSuccess);
       ret = UPTKMemcpyPeer(X_d, 1, A_d, 0,Nbytes); 
       EXPECT_EQ(ret, UPTKSuccess);
       ret = UPTKMemcpy(C_h, X_d, Nbytes, UPTKMemcpyDeviceToHost);
       EXPECT_EQ(ret, UPTKSuccess);
       UPTKDeviceSynchronize();
        
        for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i]) {
          std::cout<<"A_h[i]:"<<A_h[i]<<" ! = "<<C_h[i]<<"\n";
            }
        }

        UPTKFreeHost(A_h);
        UPTKFreeHost(C_h);
        UPTKFree(A_d);
        UPTKFree(X_d);
        #else
        EXPECT_EQ(ret, UPTKSuccess);
        #endif
}
