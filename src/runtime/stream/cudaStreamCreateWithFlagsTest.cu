#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaStream, cudaStreamCreateWithFlagsTest)
{
    UPTKStream_t stream;
    unsigned int flags;
    UPTKError_t ret1 = UPTKStreamCreateWithFlags(&stream, UPTKStreamDefault);
    EXPECT_EQ(ret1, UPTKSuccess);

    UPTKError_t ret2 = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret2, UPTKSuccess);


    UPTKError_t ret3 = UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking);
    EXPECT_EQ(ret3, UPTKSuccess);

    UPTKError_t ret4 = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret4, UPTKSuccess);
}