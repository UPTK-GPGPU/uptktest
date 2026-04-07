#include <iostream>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>

#define WIDTH 32

#define NUM (WIDTH * WIDTH)


using namespace std;
namespace{
cudaStream_t stream;
void OneStream(float* data, float* randArray, float* gpuMatrix, int width) {
    cudaError_t ret = cudaSuccess;
    ret = cudaStreamCreate(&stream);
    EXPECT_EQ(ret, cudaSuccess);
    cudaMalloc((void**)&data, NUM * sizeof(float));
    cudaError_t localerror;
    CUresult localerror2;
    localerror = cudaMemcpyAsync(data, randArray, NUM * sizeof(float), cudaMemcpyHostToDevice, stream);
    EXPECT_EQ(localerror, cudaSuccess);
    localerror2 = cuMemcpyDtoHAsync(gpuMatrix, (CUdeviceptr)data, NUM * sizeof(float), stream);
    EXPECT_EQ(localerror2, CUDA_SUCCESS);
}


TEST(cuMemory, cuMemcpyHtoDAsyncTest1) {
    cudaSetDevice(0);

    float *data, *gpuMatrix, *randArray;
    cudaError_t ret = cudaSuccess;
    int width = WIDTH;

    randArray = (float*)malloc(NUM * sizeof(float));
    gpuMatrix = (float*)malloc(NUM * sizeof(float));

    for (int i = 0; i < NUM; i++) {
        randArray[i] = (float)i * 1.0f;
    }

    OneStream(data, randArray, gpuMatrix, width);
    cudaDeviceSynchronize();

    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        EXPECT_LT(std::abs(randArray[i] - gpuMatrix[i]), eps)  << i << " before: " << randArray[i] <<  " after: " << gpuMatrix[i];
        if (std::abs(randArray[i] - gpuMatrix[i]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);


    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(ret, cudaSuccess);
    free(randArray);
    free(gpuMatrix);
    cudaFree(data);
    cudaDeviceReset();
}
}
