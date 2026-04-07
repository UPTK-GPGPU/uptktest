
#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


#define WIDTH 1024
#define N (WIDTH * WIDTH)

#define STREAM_NUM 4

void MatrixCPUAddition_08(int* a, int *b, int*c) {
    for(int i=0;i<N;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}


__global__ void MatrixAddition_08(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

TEST(cudaStream, cudaMultiStreamTest_4){

    int streamSize = N/STREAM_NUM;
    int streamBytes = streamSize * sizeof(int);

    UPTKError_t ret = UPTKSuccess;
    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;
    int *Result_h;
    size_t Nbytes = N * sizeof(int);

    A_h = (int *)malloc(Nbytes);
    EXPECT_TRUE(A_h!=NULL);
    B_h = (int *)malloc(Nbytes);
    EXPECT_TRUE(B_h!=NULL);
    C_h = (int *)malloc(Nbytes);
    EXPECT_TRUE(C_h!=NULL);
    Result_h = (int *)malloc(Nbytes);
    EXPECT_TRUE(Result_h!=NULL);

    for (int i = 0; i < N; i++)  
    {  
        A_h[i] = i;  
        B_h[i] = N - i;  
    }

    ret = UPTKMalloc(&A_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&B_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&C_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKStream_t mystream[STREAM_NUM];
    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }


    UPTKMemcpyAsync(&A_d[0], &A_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[0]);
    UPTKMemcpyAsync(&B_d[0], &B_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[0]);
    ///hipLaunchKernelGGL(MatrixAddition_08, streamSize/WIDTH , WIDTH, 0 , mystream[0] , &A_d[0], &B_d[0], &C_d[0]);
    MatrixAddition_08<<<streamSize/WIDTH , WIDTH, 0 , mystream[0]>>> (&A_d[0], &B_d[0], &C_d[0]);

    UPTKStreamSynchronize(mystream[0]);
    UPTKMemcpyAsync(&C_h[0] , &C_d[0], streamBytes, UPTKMemcpyDeviceToHost,mystream[0]);

    UPTKMemcpyAsync(&A_d[streamSize], &A_h[streamSize] , streamBytes ,UPTKMemcpyHostToDevice,mystream[1]);
    UPTKMemcpyAsync(&B_d[streamSize], &B_h[streamSize] , streamBytes ,UPTKMemcpyHostToDevice,mystream[1]);
    ///hipLaunchKernelGGL(MatrixAddition_08, streamSize/WIDTH , WIDTH, 0 , mystream[1] , &A_d[streamSize], &B_d[streamSize], &C_d[streamSize]);
    MatrixAddition_08<<<streamSize/WIDTH , WIDTH, 0 , mystream[1]>>>(&A_d[streamSize], &B_d[streamSize], &C_d[streamSize]);

    UPTKStreamSynchronize(mystream[1]);
    UPTKMemcpyAsync(&C_h[streamSize] , &C_d[streamSize], streamBytes, UPTKMemcpyDeviceToHost,mystream[1]);

    UPTKMemcpyAsync(&A_d[streamSize*2], &A_h[streamSize*2] , streamBytes ,UPTKMemcpyHostToDevice,mystream[2]);
    UPTKMemcpyAsync(&B_d[streamSize*2], &B_h[streamSize*2] , streamBytes ,UPTKMemcpyHostToDevice,mystream[2]);
    ///hipLaunchKernelGGL(MatrixAddition_08, streamSize/WIDTH , WIDTH, 0 , mystream[2] , &A_d[streamSize*2], &B_d[streamSize*2], &C_d[streamSize*2]);
    MatrixAddition_08<<<streamSize/WIDTH , WIDTH, 0 , mystream[2]>>> (&A_d[streamSize*2], &B_d[streamSize*2], &C_d[streamSize*2]);

    UPTKStreamSynchronize(mystream[2]);
    UPTKMemcpyAsync(&C_h[streamSize*2] , &C_d[streamSize*2], streamBytes, UPTKMemcpyDeviceToHost,mystream[2]);

    UPTKMemcpyAsync(&A_d[streamSize*3], &A_h[streamSize*3] , streamBytes ,UPTKMemcpyHostToDevice,mystream[3]);
    UPTKMemcpyAsync(&B_d[streamSize*3], &B_h[streamSize*3] , streamBytes ,UPTKMemcpyHostToDevice,mystream[3]);
    ///hipLaunchKernelGGL(MatrixAddition_08, streamSize/WIDTH , WIDTH, 0 , mystream[3] , &A_d[streamSize*3], &B_d[streamSize*3], &C_d[streamSize*3]);
    MatrixAddition_08<<<streamSize/WIDTH , WIDTH, 0 , mystream[3]>>>(&A_d[streamSize*3], &B_d[streamSize*3], &C_d[streamSize*3]);

    UPTKStreamSynchronize(mystream[3]);
    UPTKMemcpyAsync(&C_h[streamSize*3] , &C_d[streamSize*3], streamBytes, UPTKMemcpyDeviceToHost,mystream[3]);

    UPTKStreamSynchronize(NULL);
    MatrixCPUAddition_08(A_h,B_h,Result_h);
    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < N; i++) {
        if (std::abs(C_h[i] - Result_h[i]) > eps) {
            errors++;
        }

    }
    if (errors != 0) {
        EXPECT_EQ(errors,0);
        printf("FAILED: %d errors\n", errors);
    }
    else {
        printf("PASSED!\n");
    }

    for(int i=0;i<STREAM_NUM;i++){
       ret = UPTKStreamDestroy(mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess);
    }


    
    free(A_h);
    free(B_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(C_d);
    UPTKFree(Result_h);

}
