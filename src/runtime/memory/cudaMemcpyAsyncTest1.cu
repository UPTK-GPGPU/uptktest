#include <iostream>
#include <cuda.h>
#include <gtest/gtest.h>
#include <stdio.h>

#define WIDTH 32

#define NUM (WIDTH * WIDTH)

using namespace std;

TEST(cudaMemory, cudaMemcpyAsyncTest1) {
    UPTKSetDevice(0);

    int numDevice = 0;
    float *buffer = NULL;
    float *gpuMatrix = NULL;
    int bufferSize = NUM * sizeof(float);
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKMallocHost(&buffer,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMallocHost(&gpuMatrix,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((float *)attribs.hostPointer,buffer);//attribs.devicePointer=0x7fd5f8e00000,buffer=0x7fd5f8e00000
   
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = UPTKPointerGetAttributes(&attribs,(void *)gpuMatrix);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((float *)attribs.hostPointer,gpuMatrix);

    float *data;

    int width = WIDTH;


    for (int i = 0; i < NUM; i++) {
        buffer[i] = (float)i * 1.0f;
    }


    UPTKMalloc((void**)&data, NUM * sizeof(float));
    UPTKMemcpyAsync(data, buffer, NUM * sizeof(float), UPTKMemcpyHostToDevice, NULL);//种类测试，这个怎么搞？固定内存怎么搞？
    UPTKMemcpyAsync(gpuMatrix, data, NUM * sizeof(float), UPTKMemcpyDeviceToHost, NULL);
    UPTKDeviceSynchronize();


    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        EXPECT_LT(std::abs(buffer[i] - gpuMatrix[i]), eps)  << i << " before: " << buffer[i] <<  " after: " << gpuMatrix[i];
        if (std::abs(buffer[i] - gpuMatrix[i]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);
    
    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKFree(data);
    ret = UPTKFree(gpuMatrix);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKDeviceReset();
}