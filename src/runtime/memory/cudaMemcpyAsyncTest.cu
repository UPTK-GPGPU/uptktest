#include <iostream>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>

#define WIDTH 32

#define NUM (WIDTH * WIDTH)


using namespace std;
UPTKStream_t stream;
void OneStream(float* data, float* randArray, float* gpuMatrix, int width) {
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&stream);
    EXPECT_EQ(ret, UPTKSuccess);		
    UPTKMalloc((void**)&data, NUM * sizeof(float));
    UPTKError_t localerror;
    localerror = UPTKMemcpyAsync(data, randArray, NUM * sizeof(float), UPTKMemcpyHostToDevice, stream);
    EXPECT_EQ(localerror, UPTKSuccess);
    localerror = UPTKMemcpyAsync(gpuMatrix, data, NUM * sizeof(float), UPTKMemcpyDeviceToHost, stream);
    EXPECT_EQ(localerror, UPTKSuccess);
}


TEST(cudaMemory, cudaMemcpyAsyncTest) {
    UPTKSetDevice(0);

    float *data, *gpuMatrix, *randArray;

    int width = WIDTH;
    UPTKError_t ret = UPTKSuccess;
    randArray = (float*)malloc(NUM * sizeof(float));
    gpuMatrix = (float*)malloc(NUM * sizeof(float));

    for (int i = 0; i < NUM; i++) {
        randArray[i] = (float)i * 1.0f;
    }

    OneStream(data, randArray, gpuMatrix, width);
    UPTKDeviceSynchronize();

    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        EXPECT_LT(std::abs(randArray[i] - gpuMatrix[i]), eps)  << i << " before: " << randArray[i] <<  " after: " << gpuMatrix[i];
        if (std::abs(randArray[i] - gpuMatrix[i]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);

    ret = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret, UPTKSuccess);
    free(randArray);
    free(gpuMatrix);
    UPTKFree(data);
    UPTKDeviceReset();
}
