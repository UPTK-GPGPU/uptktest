
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

void MatrixCPUAddition_20(int* a, int *b, int*c) {
    for(int i=0;i<N;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}


__global__ void MatrixAddition_20(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

__global__ void MatrixAddition_zero20(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = 0;
    }
}


TEST(cudaStream, cudaMultiStreamTest_6){

    int streamSize = N;
    int streamBytes = streamSize * sizeof(int);
    int cal_size = N/1024;
    int cal_Bytes = cal_size * sizeof(int);

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

    UPTKStream_t mystream;
    ret = UPTKStreamCreate(&mystream);
    EXPECT_EQ(ret, UPTKSuccess);


    UPTKMemcpyAsync(A_d, A_h , streamBytes ,UPTKMemcpyHostToDevice,mystream);
    UPTKMemcpyAsync(B_d, B_h , streamBytes ,UPTKMemcpyHostToDevice,mystream);
    for(int i =0;i<1024;i++){
        ///hipLaunchKernelGGL(MatrixAddition_20, cal_size/WIDTH , WIDTH, 0, mystream, &A_d[i*cal_size], &B_d[i*cal_size], &C_d[i*cal_size]);
        MatrixAddition_20<<<cal_size/WIDTH , WIDTH, 0, mystream>>>(&A_d[i*cal_size], &B_d[i*cal_size], &C_d[i*cal_size]);

    }
    UPTKMemcpyAsync(C_h , C_d, streamBytes, UPTKMemcpyDeviceToHost,mystream);


    UPTKStreamSynchronize(mystream);
    MatrixCPUAddition_20(A_h,B_h,Result_h);
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


    ret = UPTKStreamDestroy(mystream);
    EXPECT_EQ(ret, UPTKSuccess);


    
    free(A_h);
    free(B_h);
    free(C_h);
    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(C_d);
    UPTKFree(Result_h);

}
