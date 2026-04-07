#include <stdio.h>
#include <stddef.h>
#include <iostream>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
//#define N 1024*1024*1024                       //Expand the N

TEST(cudaP2P,cudaMemcpyPeerAsyncTest2)
{
    UPTKError_t ret = UPTKSuccess;
    #ifdef __TEST_HIPHSA__
    const long long N = 1073741824;
    size_t Nbytes = N * sizeof(int);
    int numDevices = 0;
    int *A_d, *B_d, *X_d, *Y_d;
    int *A_h, *B_h, *C_h,*D_h;
    UPTKStream_t s;
    
    ret = UPTKGetDeviceCount(&numDevices);
    std::cout<<numDevices;
    if (numDevices <=1)
    {
       EXPECT_EQ(ret, UPTKSuccess);
       return ;
    }
    UPTKMallocHost(&A_h, Nbytes);
    UPTKMallocHost(&B_h, Nbytes);
    UPTKMallocHost(&C_h, Nbytes);
    UPTKMallocHost(&D_h, Nbytes);
    UPTKSetDevice(0);
    UPTKMalloc(&A_d, Nbytes);
    UPTKMalloc(&B_d, Nbytes);
    ret = UPTKStreamCreate(&s);
    EXPECT_EQ(ret, UPTKSuccess);
    for(long long i = 0;i<N;i++){
        A_h[i] = 1;
        B_h[i] = 1;
    }
     
    ret = UPTKMemcpy(A_d, A_h, Nbytes, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(B_d, B_h, Nbytes, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKSetDevice(1);
    UPTKMalloc(&X_d, Nbytes);
    UPTKMalloc(&Y_d, Nbytes);
    ret = UPTKMemcpyPeerAsync(X_d,1,A_d,0,Nbytes,s);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyPeerAsync(Y_d,1,B_d,0,Nbytes,s);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(C_h, X_d, Nbytes, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(D_h, Y_d, Nbytes, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess);
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i]) {
            std::cout<<"A_h[i]:"<<A_h[i]<<" ! = "<<C_h[i]<<"\n";
            }
        }
    for (size_t i = 0; i < N; i++) {
        if (D_h[i] != B_h[i]) {
            std::cout<<"B_h[i]:"<<B_h[i]<<" ! = "<<D_h[i]<<"\n";
        }
    }

    ret = UPTKStreamDestroy(s);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKFreeHost(A_h);
    UPTKFreeHost(B_h);
    UPTKFreeHost(C_h);
    UPTKFreeHost(D_h);
    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(X_d);
    UPTKFree(Y_d);
    #else
    EXPECT_EQ(ret, UPTKSuccess);
    #endif
}
