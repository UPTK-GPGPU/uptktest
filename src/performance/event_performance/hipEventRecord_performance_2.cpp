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
#include <sys/time.h>
// hip header file
#include "cuda.h"

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose_2(float* out, float* in, const int width) {
    int x = blockDim.x *  blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference_2(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(hipPerformanceEvent,hipEventRecord_Performance_2){
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;
    struct timeval startTM1;
    struct timeval startTM;
    struct timeval endTM1;
    struct timeval endTM;
    struct timeval startTM2;
    struct timeval endTM2;
    struct timeval startTM3;
    struct timeval endTM3;
    struct timeval startTM4;
    struct timeval endTM4;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
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
    cudaMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    cudaMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Record the start event
    gettimeofday(&startTM1,NULL);
    cudaEventRecord(start, NULL);
    //gettimeofday(&endTM1,NULL);

    //gettimeofday(&startTM,NULL);
    // Memory transfer from host to device
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);
    //gettimeofday(&endTM,NULL);
    // Record the stop event
    //gettimeofday(&endTM1,NULL);
    cudaEventRecord(stop, NULL);
    gettimeofday(&startTM2,NULL);

    cudaEventSynchronize(stop);
    gettimeofday(&endTM1,NULL);
    gettimeofday(&endTM2,NULL);

    cudaEventElapsedTime(&eventMs, start, stop);
    //gettimeofday(&endTM,NULL);
    long long diff = ((endTM1.tv_sec-startTM1.tv_sec)*1000000+(endTM1.tv_usec-startTM1.tv_usec));
    std::cout << "cudaEventRecord :" << (double)diff<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;
    long long diff3 = ((endTM2.tv_sec-startTM2.tv_sec)*1000000+(endTM2.tv_usec-startTM2.tv_usec));
    std::cout << "cudaEventSynchronize :" << (double)diff3<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;
    //long long diff2 = ((endTM.tv_sec-startTM.tv_sec)*1000000+(endTM.tv_usec-startTM.tv_usec));
    //std::cout << "The cost time :" << (double)diff2<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;

    printf("cudaMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);
    
    //hipLaunchKernelGGL( matrixTranspose_2, 
    //                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
    //                    0,
    //                    0, 
    //                    gpuTransposeMatrix,
    //                    gpuMatrix, 
    //                    WIDTH);
    matrixTranspose_2<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>
                        (gpuTransposeMatrix, gpuMatrix, WIDTH);
    // Record the start event
    cudaEventRecord(start, NULL);
    gettimeofday(&startTM,NULL);
    gettimeofday(&startTM4,NULL);

    // Lauching kernel from host
    //hipLaunchKernelGGL( matrixTranspose_2, 
    //                    dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
    //                    0,
    //                    0, 
    //                    gpuTransposeMatrix,
    //                    gpuMatrix, 
    //                    WIDTH);
    matrixTranspose_2<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>
                        (gpuTransposeMatrix, gpuMatrix, WIDTH);
    // Record the stop event
    cudaEventRecord(stop, NULL);
    gettimeofday(&startTM3,NULL);
    cudaEventSynchronize(stop);
    gettimeofday(&endTM,NULL);
    gettimeofday(&endTM3,NULL);

    cudaEventElapsedTime(&eventMs, start, stop);
    long long diff5 = ((endTM4.tv_sec-startTM4.tv_sec)*1000000+(endTM4.tv_usec-startTM4.tv_usec));
    std::cout << "hipLaunchKernelGGL :" << (double)diff5<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;
    long long diff4 = ((endTM3.tv_sec-startTM3.tv_sec)*1000000+(endTM3.tv_usec-startTM3.tv_usec));
    std::cout << "cudaEventSynchronize :" << (double)diff4<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;
    long long diff2 = ((endTM.tv_sec-startTM.tv_sec)*1000000+(endTM.tv_usec-startTM.tv_usec));
    std::cout << "all the the cost time :" << (double)diff2<<"(us), size: "<<(NUM * sizeof(float))<< std::endl;
    

    printf("kernel Execution time             = %6.3fms\n", eventMs);

    // Record the start event
    cudaEventRecord(start, NULL);

    // Memory transfer from device to host
    cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&eventMs, start, stop);

    printf("cudaMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference_2(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
            EXPECT_EQ(errors,0);
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }
    // free the resources on device side
    cudaFree(gpuMatrix);
    cudaFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

}
