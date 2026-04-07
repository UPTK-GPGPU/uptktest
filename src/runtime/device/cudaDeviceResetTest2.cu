#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 32
#define NUM (WIDTH * WIDTH)
#define SIZE 2048
# define N 32
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

__global__ void matrixTranspose(float* out, float* in, const int width) {
    __shared__ float sharedMem[WIDTH * WIDTH];

    int x = blockDim.x *  blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    sharedMem[y * width + x] = in[x * width + y];

    __syncthreads();

    out[y * width + x] = sharedMem[y * width + x];
}

TEST(cudaDevice, cudaDeviceResetTest2){
 //add kernel test case
    float* Matrix;
    float* gpuMatrix;
    float* gpuTransposeMatrix;
    UPTKError_t ret = UPTKSuccess;

    Matrix = (float*)malloc(NUM * sizeof(float));
    for (int i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }
    UPTKMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    UPTKMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    //hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
    //                gpuMatrix, WIDTH);
    matrixTranspose <<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0>>>(
                    gpuTransposeMatrix, gpuMatrix, WIDTH);
 
    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess);
   
    //hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //                dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
    //                gpuMatrix, WIDTH);
    matrixTranspose<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(
                    gpuTransposeMatrix, gpuMatrix, WIDTH);
    std::cout<<"success";

    //add multi_kernel
   //kernel 1
    int device_count = 0;
    UPTKGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; ++i) {
        UPTKSetDevice(i);
        //hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
        //            dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
        //            gpuMatrix, WIDTH);
        matrixTranspose<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(
                    gpuTransposeMatrix, gpuMatrix, WIDTH);
        UPTKDeviceSynchronize();
        ret = UPTKDeviceReset();
        EXPECT_EQ(ret, UPTKSuccess);
    }

    UPTKFree(gpuMatrix);
    UPTKFree(gpuTransposeMatrix);
    free(Matrix); 
}
