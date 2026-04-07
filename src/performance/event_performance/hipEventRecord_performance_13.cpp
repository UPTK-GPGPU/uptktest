#include <iostream>

// hip header file
#include "cuda.h"
#include <gtest/gtest.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1


__global__ void matrixTranspose_18() {
}


TEST(hipPerformanceEvent,hipEventRecord_Performance_13){
    float* Matrix;
    float* cpuMatrix;
    float* TransposeMatrix;


    float* gpuMatrix;
    int i=0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    cudaEvent_t start, stop;
    //cudaEvent_t stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaEventCreate(&stop1);
    float eventMs = 1.0f;

    Matrix = (float*)malloc(NUM * sizeof(float));
    cpuMatrix = (float*)malloc(NUM * sizeof(float));
    //TransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    cudaMalloc((void**)&gpuMatrix, NUM * sizeof(float));

    // DO NOTHING
    // Record the start event
    cudaEventRecord(start, NULL);
    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);
    printf("Do Nothing  = %6.3fms\n", eventMs);
    //EXPECT_LE(eventMs,20);

    // DO NOTHING with STOP TWICE
    // Record the start event
    cudaEventRecord(start, NULL);
    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);
    printf("Do Nothing with the first stop point = %6.3fms\n", eventMs);
    //EXPECT_LE(eventMs,20);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);
    printf("Do Nothing with the second stop point = %6.3fms\n", eventMs);
    //cudaEventRecord(stop1, NULL);
    //cudaEventSynchronize(stop1);
    //cudaEventElapsedTime(&eventMs, start, stop1);
    //printf("Do Nothing with new stop point  = %6.3fms\n", eventMs);
    

    //EVENT RECORD WITH A COPY from Host to Device
    // Record the start event
    cudaEventRecord(start, NULL);
    // Memory transfer from host to device
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);
    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);
    printf("cudaMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    // Record the start event
    cudaEventRecord(start, NULL);
    // Memory transfer from device to host
    cudaMemcpy(cpuMatrix, gpuMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);
    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);\
    printf("cudaMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);


    // Record the start event
    cudaEventRecord(start, NULL);
    // Memory transfer from device to host
    //hipLaunchKernelGGL(matrixTranspose_18, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0);
    matrixTranspose_18<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>();

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&eventMs, start, stop);\
    printf("Execute an empty kenel time taken  = %6.3fms\n", eventMs);


    // free the resources on device side
    cudaFree(gpuMatrix);
    // free the resources on host side
    free(Matrix);
    free(cpuMatrix);
    //free(TransposeMatrix);
}
