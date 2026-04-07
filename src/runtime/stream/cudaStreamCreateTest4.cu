
#include <iostream>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose_8(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}


// CPU implementation of matrix transpose
void matrixTransposeCPUReference_8(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(cudaStream, cudaStreamCreateTest4) {
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;
    
    UPTKError_t ret = UPTKSuccess;
    UPTKDeviceProp devProp;
    UPTKGetDeviceProperties(&devProp, 0);

    UPTKStream_t pstream;
    ret = UPTKStreamCreate(&pstream);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKEvent_t start, stop;
    UPTKEventCreate(&start);
    UPTKEventCreate(&stop);
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
    UPTKMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    UPTKMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
    
    // Record the start event
    UPTKEventRecord(start, NULL);

    // Memory transfer from host to device
    UPTKMemcpyAsync(gpuMatrix, Matrix, NUM * sizeof(float), UPTKMemcpyHostToDevice, pstream);

    // Record the stop event
    UPTKEventRecord(stop, NULL);
    UPTKEventSynchronize(stop);

    UPTKEventElapsedTime(&eventMs, start, stop);

    printf("UPTKMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);

    // Record the start event
    UPTKEventRecord(start, NULL);

    // Lauching kernel from host
    //hipLaunchKernelGGL(matrixTranspose_8, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstream , gpuTransposeMatrix, gpuMatrix, WIDTH);
    matrixTranspose_8<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstream>>>(gpuTransposeMatrix, gpuMatrix, WIDTH);

    // Record the stop event
    UPTKEventRecord(stop, NULL);
    UPTKEventSynchronize(stop);

    UPTKEventElapsedTime(&eventMs, start, stop);

    printf("kernel Execution time             = %6.3fms\n", eventMs);


    // Record the start event
    UPTKEventRecord(start, NULL);
    UPTKMemcpyAsync(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), UPTKMemcpyDeviceToHost, pstream);
    UPTKStreamSynchronize(pstream);

    // Record the stop event
    UPTKEventRecord(stop, NULL);
    UPTKEventSynchronize(stop);

    UPTKEventElapsedTime(&eventMs, start, stop);

    printf("UPTKMemcpyDeviceToHost time taken  = %6.3fms\n", eventMs);

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference_8(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        EXPECT_EQ(errors, 0);
        printf("FAILED: %d errors\n", errors);
    }
    else {
        printf("PASSED!\n");
    }

    ret = UPTKStreamDestroy(pstream);
    EXPECT_EQ(ret, UPTKSuccess);
    // free the resources on device side
    UPTKFree(gpuMatrix);
    UPTKFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

}
