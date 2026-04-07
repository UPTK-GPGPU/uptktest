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
__global__ void matrixTranspose_13(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}


// CPU implementation of matrix transpose
void matrixTransposeCPUReference_13(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(hipPerformanceEvent,hipEventRecord_Performance_10){
    cudaError_t ret = cudaSuccess;
    
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;
	
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    cudaStream_t pstream;
    ret = cudaStreamCreate(&pstream);
    //EXPECT_EQ(ret, cudaSuccess);
    if(ret !=cudaSuccess)
    {
        EXPECT_EQ(cudaSuccess, CUDA_SUCCESS);
        return ;
    }
    EXPECT_EQ(ret, cudaSuccess);
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float eventMs = 1.0f;
	
	int i;
    int errors;
	
    std::cout << "Device name " << devProp.name << std::endl;
	
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
	

    for(int i=0; i < 1000; i++){
    // Record the start event
    cudaEventRecord(start, NULL);

    // Memory transfer from host to device
    cudaMemcpyAsync(gpuMatrix, Matrix, NUM * sizeof(float), cudaMemcpyHostToDevice,pstream);

    // Lauching kernel from host
    //hipLaunchKernelGGL(matrixTranspose_13, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstream, gpuTransposeMatrix, gpuMatrix, WIDTH);
    matrixTranspose_13<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                      dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstream>>>
                      (gpuTransposeMatrix, gpuMatrix, WIDTH);
    cudaMemcpyAsync(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), cudaMemcpyDeviceToHost,pstream);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&eventMs, start, stop);

    printf("%d   kernel Execution time             = %6.3fms\n", i,eventMs);



}
	
	cudaStreamSynchronize(pstream);
	
	// CPU MatrixTranspose computation
    matrixTransposeCPUReference_13(cpuTransposeMatrix, Matrix, WIDTH);
	
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
	
    ret = cudaStreamDestroy(pstream);
    EXPECT_EQ(ret, cudaSuccess);	
    // free the resources on device side
    cudaFree(gpuMatrix);
    cudaFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

}
