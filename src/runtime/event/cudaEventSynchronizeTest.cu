#include <iostream>
#include <gtest/gtest.h>  

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

TEST(cudaEvent, cudaEventSynchronizeTest){
    UPTKEvent_t start, stop;
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess);
    float eventMs = 1.0f;
    float *Matrix, *gpuMatrix;

    Matrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (int i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    ret = UPTKMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    EXPECT_EQ(ret, UPTKSuccess);

     // Record the start event 
    ret =  UPTKEventRecord(start, NULL);
    EXPECT_EQ(ret, UPTKSuccess);
    // Memory transfer from host to device
    UPTKMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), UPTKMemcpyHostToDevice);

    // Record the stop event
    ret =  UPTKEventRecord(stop, NULL);
    EXPECT_EQ(ret, UPTKSuccess);
    ret =  UPTKEventSynchronize(stop);
    EXPECT_EQ(ret, UPTKSuccess);
}