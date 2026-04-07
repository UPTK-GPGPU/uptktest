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

#define repeat 100

void MatrixCPUAddition_10(int* a, int *b, int*c) {
    for(int i=0;i<N;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}


__global__ void MatrixAddition_10(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

__global__ void MatrixAddition_zero10(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = 0;
    }
}


TEST(cudaStream, cudaMultiDevices_1){


    int streamSize = N;
    int streamBytes = streamSize * sizeof(int);

    UPTKError_t ret = UPTKSuccess;
    int *A_d, *B_d, *C_d;
    int *A_h, *B_h, *C_h;
    int *Result_h;
    size_t Nbytes = N * sizeof(int);

    int errors = 0;
    double eps = 1.0E-6;


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


    int numDevices = 0;
    int STREAM_NUM = 0;
    UPTKGetDeviceCount(&numDevices);
    if(numDevices!=4){
        printf("Pass \n");
        return;
    }
    //printf("num of devices : %d \n", numDevices);
    //for(int i = 0;;)
    STREAM_NUM = numDevices * 8;
    MatrixCPUAddition_10(A_h,B_h,Result_h);


    UPTKStream_t mystream[STREAM_NUM];
    for(int i=0;i<STREAM_NUM;i++){
      ret= UPTKSetDevice(STREAM_NUM%8);
      EXPECT_EQ(ret, UPTKSuccess);
      ret = UPTKStreamCreate(&mystream[i]);
      EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
   }
    for(int i = 0; i< STREAM_NUM ;i++){
        ret= UPTKSetDevice(STREAM_NUM % 8);
        EXPECT_EQ(ret, UPTKSuccess);
        UPTKMemcpyAsync(A_d, &A_h[0] , streamBytes ,UPTKMemcpyHostToDevice);
        UPTKMemcpyAsync(B_d, &B_h[0] , streamBytes ,UPTKMemcpyHostToDevice);
        //hipLaunchKernelGGL(MatrixAddition_10, streamSize/WIDTH , WIDTH, 0 , 0, A_d, B_d, C_d);
        MatrixAddition_10<<<streamSize/WIDTH , WIDTH>>> (A_d, B_d, C_d);

        UPTKMemcpyAsync(&C_h[0] , C_d, streamBytes, UPTKMemcpyDeviceToHost);
    }
    errors = 0;
    eps = 1.0E-6;
    for(int j=1;j<repeat;j++){
        for(int i=0;i<STREAM_NUM;i++){
            ret= UPTKSetDevice(STREAM_NUM % 8);
            EXPECT_EQ(ret, UPTKSuccess);
            //hipLaunchKernelGGL(MatrixAddition_zero10, streamSize/WIDTH , WIDTH, 0, mystream[i] , A_d, B_d, C_d);
            MatrixAddition_zero10<<<streamSize/WIDTH , WIDTH, 0, mystream[i]>>> (A_d, B_d, C_d);

            UPTKMemcpyAsync(A_d, &A_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[i]);
            UPTKMemcpyAsync(B_d, &B_h[0] , streamBytes ,UPTKMemcpyHostToDevice,mystream[i]);
            //hipLaunchKernelGGL(MatrixAddition_10, streamSize/WIDTH , WIDTH, 0, mystream[i] , A_d, B_d, C_d);
            MatrixAddition_10<<<streamSize/WIDTH , WIDTH, 0, mystream[i]>>> (A_d, B_d, C_d);
            UPTKMemcpyAsync(&C_h[0] , C_d, streamBytes, UPTKMemcpyDeviceToHost,mystream[i]);
        }
        UPTKStreamSynchronize(NULL);
        if(j%10==0){
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
            }
    }


   for(int i=0;i<STREAM_NUM;i++){
      ret = UPTKStreamDestroy(mystream[i]);
      EXPECT_EQ(ret, UPTKSuccess)<< "create stream failed,index:"<<i;
   }


    free(A_h);
    free(B_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(C_d);
    UPTKFree(Result_h);

}
