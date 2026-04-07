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
__global__ void matrixTranspose_18(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

__global__ void matrixTranspose_17() {
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference_17(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(hipPerformanceEvent,hipEventRecord_Performance_17) {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;
    float time_use=0;
    struct timeval startTime;
    struct timeval endTime;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float eventMs = 1.0f;
    cudaError_t ret = cudaSuccess;

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
    cudaEventRecord(start, NULL);

    // Memory transfer from host to device
    cudaMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    ret = cudaEventQuery(stop); 
    int loop = 0;
    while(ret == cudaErrorNotReady){
       sched_yield();
       if (loop> 1000){
         EXPECT_EQ(1, 0);
         return ;
       }
       ret = cudaEventQuery(stop); 
    }
    //cudaEventSynchronize(stop);

    cudaEventElapsedTime(&eventMs, start, stop);

    printf("cudaMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);
   loop = 0; 
    for(int i=0; i < 10000; i++){
    // Record the start event
    #if 1 
    if (0 == loop)
    { 
        cudaEventRecord(start, NULL);
        //gettimeofday(&startTime,NULL);
    }
    #endif
    // Lauching kernel from host
    //hipLaunchKernelGGL(matrixTranspose_18, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,gpuMatrix,WIDTH);
    matrixTranspose_18<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>> (gpuTransposeMatrix,gpuMatrix,WIDTH);

    loop++;
    //hipLaunchKernelGGL(matrixTranspose_17, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0);

    // Record the stop event
    #if 1 
    if (200 == loop)
    {
        cudaEventRecord(stop, NULL);
        //cudaEventSynchronize(stop);
        ret = cudaEventQuery(stop); 
        while(ret == cudaErrorNotReady){
           sched_yield();
           ret = cudaEventQuery(stop); 
        }

        ret = cudaEventElapsedTime(&eventMs, start, stop);
        printf("kernel Execution time             = %.10f    ,index=%d, ret=%d\n",eventMs*1000,i,ret);
        //gettimeofday(&endTime,NULL);
        //time_use=(endTime.tv_sec-startTime.tv_sec)*1000000+(endTime.tv_usec-startTime.tv_usec);

        //printf("kernel Execution time             = %.10f    ,index=%d\n",time_use,i);
        loop = 0;
     }
     #endif
}
    // Record the start event
    cudaEventRecord(start, NULL);

    // Memory transfer from device to host
    cudaMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost);
#if 0
    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&eventMs, start, stop);

    printf("cudaMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

    // CPU MatrixTranspose computation
    //matrixTransposeCPUReference_17(cpuTransposeMatrix, Matrix, WIDTH);
    matrixTransposeCPUReference_17(TransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    EXPECT_EQ(errors, 0);
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }
#endif
    // free the resources on device side
    cudaFree(gpuMatrix);
    cudaFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

}


