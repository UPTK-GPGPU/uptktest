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
__global__ void matrixTranspose_12(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}


// CPU implementation of matrix transpose
void matrixTransposeCPUReference_12(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(hipPerformanceEvent,hipEventRecord_Performance_9) {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

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

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference_12(cpuTransposeMatrix, Matrix, WIDTH);

    // allocate the memory on the device side
    cudaMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    cudaMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Memory transfer from host to device
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);
    
    errors = 0;
    double eps = 1.0E-6;
    for(int outloop=0; outloop < 1000; outloop++){
    // Record the start event
    cudaEventRecord(start, NULL);
        
    // Memory transfer from host to device
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lauching kernel from host
    //hipLaunchKernelGGL(matrixTranspose_6, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, 0);

    ///hipLaunchKernelGGL(matrixTranspose_12, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix, gpuMatrix, WIDTH);
    matrixTranspose_12<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Memory transfer from device to host
    cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&eventMs, start, stop);

    printf("%d   kernel Execution time             = %6.3fms\n",outloop, eventMs);

        // verify the results
    for (int inloop = 0; inloop < NUM; inloop++) {
        if (std::abs(TransposeMatrix[inloop] - cpuTransposeMatrix[inloop]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);
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
