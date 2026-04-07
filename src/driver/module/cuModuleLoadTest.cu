#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <gtest/gtest.h>

#define LEN 64
#define SIZE LEN << 2

#define fileName "../src/driver/module/vcpy_kernel.code"
#define kernel_name "hello_world"

TEST(cuModule,cuModuleLoadTest){
    float *A, *B;
    CUdeviceptr Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }
    cudaError_t ret = cudaSuccess;
    CUresult ret1 = CUDA_SUCCESS;
    ret1 = cuInit(0);
    EXPECT_EQ(ret1,CUDA_SUCCESS);

    CUdevice device;
    CUcontext context;
    ret1 = cuDeviceGet(&device, 0);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret1 = cuCtxCreate(&context, 0, device);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    ret1 = cuMemAlloc(&Ad, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret1 = cuMemAlloc(&Bd, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret1 = cuMemcpyHtoD(Ad, A, SIZE);
    EXPECT_EQ(ret1,CUDA_SUCCESS);
    ret1 = cuMemcpyHtoD(Bd, B, SIZE);
    EXPECT_EQ(ret1,CUDA_SUCCESS);

    CUmodule Module;
    CUfunction Function;
    ret1 = cuModuleLoad(&Module, fileName);
    EXPECT_EQ(ret1,CUDA_SUCCESS);
    std::cout<<ret;
    ret1 = cuModuleGetFunction(&Function, Module, kernel_name);
    EXPECT_EQ(ret1,CUDA_SUCCESS);
    std::cout<<ret;

    CUstream stream;
    ret1 = cuStreamCreate(&stream, 0x0);
    EXPECT_EQ(ret1,CUDA_SUCCESS);

    struct {
        void* _Ad;
        void* _Bd;
    } args;
   args._Ad = (void*) Ad;
   args._Bd = (void*) Bd;
   size_t size = sizeof(args);

    void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args, CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      CU_LAUNCH_PARAM_END};
    ret1 = cuLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, nullptr, (void**)&config);
    EXPECT_EQ(ret1,CUDA_SUCCESS);

    ret1 = cuStreamDestroy(stream);
    EXPECT_EQ(ret1,CUDA_SUCCESS);

    ret1 = cuMemcpyDtoH(B, Bd, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    for (uint32_t i = 0; i < LEN; i++) {
        //assert(A[i] == B[i]);
        EXPECT_EQ(A[i],B[i]);
    }

    ret1 = cuCtxDestroy(context);
    EXPECT_EQ(ret,CUDA_SUCCESS);
}