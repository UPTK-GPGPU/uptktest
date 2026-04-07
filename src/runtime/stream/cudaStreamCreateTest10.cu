#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>



#define N 1024

#define STREAM_NUM 31

TEST(cudaStream, cudaStreamCreateTest10){
    UPTKError_t ret = UPTKSuccess;
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t Nbytes = N * sizeof(float);

    A_h = (float *)malloc(Nbytes);
    EXPECT_TRUE(A_h!=NULL);
    C_h = (float *)malloc(Nbytes);
    EXPECT_TRUE(C_h!=NULL);

    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    ret = UPTKMalloc(&A_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&C_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKStream_t mystream[STREAM_NUM];
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }
    UPTKDeviceReset(); 
    free(A_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(C_d);
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }  
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }  
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }  
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }  
}
