#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define LEN 64
#define SIZE LEN << 2

#define FILENAME "../src/driver/module/vcpy_kernel.code"
#define kernel_name "hello_world"

TEST(cuModule,cuModuleLoadDataExTest){
    float *A, *B;
    CUdeviceptr Ad, Bd;
    A = new float[LEN];
    B = new float[LEN];

    for (uint32_t i = 0; i < LEN; i++) {
        A[i] = i * 1.0f;
        B[i] = 0.0f;
    }

   std::cout<<"start";
    
    
    CUresult ret = CUDA_SUCCESS;


    ret = cuInit(0);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret = cuMemAlloc(&Ad, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret = cuMemAlloc(&Bd, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    ret = cuMemcpyHtoD(Ad, A, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);
    ret = cuMemcpyHtoD(Bd, B, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    CUmodule Module;
    CUfunction Function;
    std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
    std::streamsize fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fsize);
    if (file.read(buffer.data(), fsize)) {
        // setup JIT compilation options and perform compilation
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        ret = cuModuleLoadDataEx(&Module, &buffer[0],jitNumOptions, jitOptions, (void **)jitOptVals);
        EXPECT_EQ(ret,CUDA_SUCCESS);
        ret = cuModuleGetFunction(&Function, Module, kernel_name);
        EXPECT_EQ(ret,CUDA_SUCCESS);
    }
    else {
        std::cout<<"could not open code object";
        std::cout<<FILENAME;
    }

    CUstream stream;
    ret = cuStreamCreate(&stream, 0x0);
    
    EXPECT_EQ(ret,CUDA_SUCCESS);

    struct {
        void* _Ad;
        void* _Bd;
    } args;
    args._Ad = (void*) Ad;
    args._Bd = (void*) Bd;
    size_t size = sizeof(args);

    void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args, CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      CU_LAUNCH_PARAM_END};
    ret = cuLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream, nullptr, (void**)&config);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    ret = cuStreamDestroy(stream);
    std::cout<<"9";
    EXPECT_EQ(ret,CUDA_SUCCESS);

    ret = cuMemcpyDtoH(B, Bd, SIZE);
    EXPECT_EQ(ret,CUDA_SUCCESS);

    for (uint32_t i = 0; i < LEN; i++) {
        //assert(A[i] == B[i]);
        EXPECT_EQ(A[i],B[i]);
    }

   ret = cuModuleUnload(Module);
   EXPECT_EQ(ret,CUDA_SUCCESS);
}