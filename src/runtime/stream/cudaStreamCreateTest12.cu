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

#define STREAM_NUM 31

void MatrixCPUAddition_02(int* a, int *b, int*c) {
    for(int i=0;i<N;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}


__global__ void MatrixAddition_02(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

TEST(cudaStream, cudaStreamCreateTest12){

    int streamSize = N;
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
    UPTKDeviceReset(); 
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
    
    for(int j=0;j<999;j++){
        for(int i=0;i<STREAM_NUM;i++){
           ret = UPTKStreamCreate(&mystream[i]);
           EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
        }
        for(int i=0;i<STREAM_NUM;i++){
        ret = UPTKStreamDestroy(mystream[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        }
    }

    for(int i=0;i<STREAM_NUM-1;i++){
       ret = UPTKStreamCreate(&mystream[i]);
       EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
    }

    ret = UPTKStreamCreate(&mystream[STREAM_NUM-1]);
    EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<STREAM_NUM-1;
    UPTKMemcpyAsync(A_d, &A_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[STREAM_NUM-1]);
    UPTKMemcpyAsync(B_d, &B_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[STREAM_NUM-1]);
    MatrixAddition_02 <<< streamSize/WIDTH , WIDTH, 0, mystream[STREAM_NUM-1] >>>(A_d, B_d, C_d);
    UPTKMemcpyAsync(&C_h[0] , C_d, streamBytes, UPTKMemcpyDeviceToHost , mystream[STREAM_NUM-1]);
    MatrixCPUAddition_02(A_h,B_h,Result_h);
    
    // verify the results
    int errors = 0;
    //double eps = 1.0E-6;
    for (int i = 0; i < N; i++) {
        if (C_h[i] != Result_h[i]) {
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
}
