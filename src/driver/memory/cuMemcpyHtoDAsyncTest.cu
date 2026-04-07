#include <iostream>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>

#define WIDTH 32

#define NUM (WIDTH * WIDTH)


using namespace std;
namespace{


TEST(cuMemory, cuMemcpyHtoDAsyncTest) {
    cudaSetDevice(0);

    int numDevice = 0;
    float *buffer = NULL;
    float *gpuMatrix = NULL;
    int bufferSize = NUM * sizeof(float);
    cudaError_t ret = cudaSuccess;
    ret = cudaMallocHost(&buffer,bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    ret = cudaMallocHost(&gpuMatrix,bufferSize);
    EXPECT_EQ(ret, cudaSuccess);

    cudaPointerAttributes attribs;
    memset(&attribs, 0, sizeof(cudaPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = cudaPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ((float *)attribs.hostPointer,buffer);//attribs.devicePointer=0x7fd5f8e00000,buffer=0x7fd5f8e00000
   
    memset(&attribs, 0, sizeof(cudaPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = cudaPointerGetAttributes(&attribs,(void *)gpuMatrix);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ((float *)attribs.hostPointer,gpuMatrix);

    float *data;

    int width = WIDTH;


    for (int i = 0; i < NUM; i++) {
        buffer[i] = (float)i * 1.0f;
    }


    cudaMalloc((void**)&data, NUM * sizeof(float));

    cuMemcpyHtoDAsync((CUdeviceptr)data, buffer, NUM * sizeof(float), NULL);
   
    memset(&attribs, 0, sizeof(cudaPointerAttributes));
    ret = cudaPointerGetAttributes(&attribs,(void *)data);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ((float *)attribs.devicePointer, data);
    
   
    cudaMemcpyAsync(gpuMatrix, data, NUM * sizeof(float), cudaMemcpyDeviceToHost, NULL);

    cudaDeviceSynchronize();

    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        EXPECT_LT(std::abs(buffer[i] - gpuMatrix[i]), eps)  << i << " before: " << buffer[i] <<  " after: " << gpuMatrix[i];
        if (std::abs(buffer[i] - gpuMatrix[i]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);

    cudaFree(data);
    ret = cudaFree(buffer);
    EXPECT_EQ(ret, cudaSuccess);
    ret = cudaFree(gpuMatrix);
    EXPECT_EQ(ret, cudaSuccess);
    cudaDeviceReset();
}
}