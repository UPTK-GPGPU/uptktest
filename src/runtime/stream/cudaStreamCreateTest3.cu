
#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>



#define N 1024

#define STREAM_NUM 8

__global__ void vector3_square(float* C_d, float* A_d, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < n) C_d[i] = A_d[i] * A_d[i];
    
}

TEST(cudaStream, cudaStreamCreateTest3){
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
    UPTKStream_t mystream2[STREAM_NUM];
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }
    
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream2[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }
    
    ret = UPTKMemcpyAsync(A_d, A_h, Nbytes, UPTKMemcpyHostToDevice, mystream[0]);
    EXPECT_EQ(ret, UPTKSuccess);

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    ///hipLaunchKernelGGL(vector3_square, dim3(blocks), dim3(threadsPerBlock), 0, mystream[0], C_d, A_d,N);
    vector3_square<<< dim3(blocks), dim3(threadsPerBlock), 0, mystream[0]>>> (C_d, A_d,N);

    ret = UPTKMemcpyAsync(C_h, C_d, Nbytes, UPTKMemcpyDeviceToHost, mystream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    
    ret = UPTKStreamSynchronize(mystream[0]);
    EXPECT_EQ(ret, UPTKSuccess);

    for (size_t i = 0; i < N; i++) {
        //if (C_h[i] != A_h[i] * A_h[i]) {
        //    printf("Data mismatch %zu", i);
        //}
        EXPECT_TRUE(C_h[i]==A_h[i] * A_h[i]) << "Data mismatch\n";
    }

    free(A_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(C_d);
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }
    
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream2[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }
}
