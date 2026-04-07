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

void MatrixCPUAddition_14(int* a, int *b, int*c) {
    for(int i=0;i<N;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}


__global__ void MatrixAddition_14(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

__global__ void MatrixAddition_zero14(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = 0;
    }
}


TEST(cudaStream, cudaMultiDevices_4){


    int numDevices = 0;
    UPTKGetDeviceCount(&numDevices);
    if(numDevices!=4){
        printf("Pass \n");
        return;
    }
    //printf("num of devices : %d \n", numDevices);

    int streamSize = N/numDevices;
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

    ret = UPTKMallocHost(&A_d, Nbytes, 0);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMallocHost(&B_d, Nbytes, 0);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMallocHost(&C_d, Nbytes, 0);
    EXPECT_EQ(ret, UPTKSuccess);



    MatrixCPUAddition_14(A_h,B_h,Result_h);

    for(int i=0;i<numDevices ;i++){
        ret= UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);
        UPTKMemcpyAsync(&A_d[streamSize*i], &A_h[streamSize*i] , streamBytes ,UPTKMemcpyHostToDevice);
        UPTKMemcpyAsync(&B_d[streamSize*i], &B_h[streamSize*i] , streamBytes ,UPTKMemcpyHostToDevice);
        MatrixAddition_14 <<< streamSize/WIDTH , WIDTH, 0 >>>(&A_d[streamSize*i], &B_d[streamSize*i], &C_d[streamSize*i]);
        //UPTKStreamSynchronize(mystream[i]);
        UPTKMemcpyAsync(&C_h[streamSize*i] , &C_d[streamSize*i], streamBytes, UPTKMemcpyDeviceToHost);
    }

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

    free(A_h);
    free(B_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(C_d);
    UPTKFree(Result_h);

    
}
