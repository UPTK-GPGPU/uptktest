#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaStream, cudaStreamCreateTest)
{
    UPTKStream_t stream;
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&stream);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKStreamDestroy(stream);
}