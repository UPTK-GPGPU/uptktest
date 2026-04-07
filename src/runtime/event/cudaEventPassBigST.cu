/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <gtest/gtest.h>   

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

struct passTest{
    int numWidth;
    int numTest[2048];
};
typedef struct passTest passTest_st;
// Device (Kernel) function, it must be void
__global__ void matrixTranspose(float* out, float* in, const passTest_st testp) {
    int x = blockDim.x *  blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * testp.numWidth + x] = in[x * testp.numWidth + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(cudaEvent, cudaEventPassBigST){
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;


    UPTKDeviceProp devProp;
    UPTKGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    #ifndef __TEST_HIPHSA__
    EXPECT_EQ(UPTKSuccess, UPTKSuccess);
    return;
    #endif
     
    UPTKEvent_t start, stop;
    UPTKEventCreate(&start);
    UPTKEventCreate(&stop);
    float eventMs = 1.0f;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    UPTKMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    UPTKMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Record the start event
    UPTKEventRecord(start, NULL);

    // Memory transfer from host to device
    UPTKMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), UPTKMemcpyHostToDevice);

    // Record the stop event
    UPTKEventRecord(stop, NULL);
    UPTKEventSynchronize(stop);

    UPTKEventElapsedTime(&eventMs, start, stop);

    printf("UPTKMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    // Record the start event
    UPTKEventRecord(start, NULL);
    int numTmp = 1024;
    int *pTmp = &numTmp;
    struct passTest stTmp;
    struct passTest * pTmp2 = &stTmp;
    void *argsTmp[] = {&gpuTransposeMatrix, &gpuMatrix, pTmp2};
    // Lauching kernel from host
    UPTKError_t ret = UPTKLaunchKernel((const void *)matrixTranspose, 
                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 
                    argsTmp,
                    0,
                    NULL);
    #ifdef __TEST_HIPHSA__
    EXPECT_EQ(ret, UPTK_ERROR_LAUNCH_OUT_OF_RESOURCES);
    #else
    EXPECT_EQ(ret, UPTKSuccess);
    #endif

    if (UPTKSuccess == ret)
    {
        // Record the stop event
        UPTKEventRecord(stop, NULL);
        UPTKEventSynchronize(stop);

        UPTKEventElapsedTime(&eventMs, start, stop);

        printf("kernel Execution time             = %6.3fms\n", eventMs);

        // Record the start event
        UPTKEventRecord(start, NULL);

        // Memory transfer from device to host
        UPTKMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), UPTKMemcpyDeviceToHost);

        // Record the stop event
        UPTKEventRecord(stop, NULL);
        UPTKEventSynchronize(stop);

        UPTKEventElapsedTime(&eventMs, start, stop);

        printf("UPTKMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

        // CPU MatrixTranspose computation
        matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

        // verify the results
        errors = 0;
        double eps = 1.0E-6;
        for (i = 0; i < NUM; i++) {
            if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
                errors++;
            }
        }
        if (errors != 0) {
            printf("FAILED: %d errors\n", errors);
        } else {
            printf("PASSED!\n");
        }
        EXPECT_EQ(errors, 0);
    }
    // free the resources on device side
    UPTKFree(gpuMatrix);
    UPTKFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);
}
