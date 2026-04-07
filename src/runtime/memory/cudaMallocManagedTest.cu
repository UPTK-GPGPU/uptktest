#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

__global__ void add(int N, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) y[i] = x[i] + y[i];
}

TEST(cudaMemory,cudaMallocManagedTest){
    int numElements = 100;
    float *A = NULL; 
    float *B = NULL;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMallocManaged(&A, numElements*sizeof(float));
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocManaged failed";
    UPTKMallocManaged(&B, numElements*sizeof(float));
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocManaged failed";

    for (int i = 0; i < numElements; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(blockSize, 1, 1);
    //cudaLaunchKernelGGL(add, dimGrid, dimBlock, 0, 0, numElements, A, B);
    add<<<dimGrid, dimBlock>>>(numElements, A, B);

    UPTKDeviceSynchronize();

    for (int i = 0; i < numElements; i++){
        EXPECT_TRUE(B[i] == 3.0f) << "Output Mismatch";
    }

    UPTKFree(A);
    UPTKFree(B);
}