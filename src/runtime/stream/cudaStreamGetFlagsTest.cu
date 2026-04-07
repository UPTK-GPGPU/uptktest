#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaStream,cudaStreamGetFlagsTest)
{
    UPTKStream_t stream;
    unsigned int flags;
    
    UPTKError_t ret = UPTKStreamCreate(&stream);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKStreamGetFlags(stream, &flags);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ(flags, 0);
    ret= UPTKStreamDestroy(stream);
    EXPECT_EQ(ret, UPTKSuccess);
    
    
    UPTKError_t ret1 = UPTKStreamCreateWithFlags(&stream, UPTKStreamDefault);
    UPTKError_t ret2 = UPTKStreamGetFlags(stream, &flags);
    EXPECT_EQ(ret1, UPTKSuccess);
    EXPECT_EQ(ret2, UPTKSuccess);
    EXPECT_EQ(flags, 0);

    UPTKError_t ret3 = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret3, UPTKSuccess);


    UPTKError_t ret4 = UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking);
    UPTKError_t ret5 = UPTKStreamGetFlags(stream, &flags);
    EXPECT_EQ(ret4, UPTKSuccess);
    EXPECT_EQ(ret5,UPTKSuccess);
    EXPECT_EQ(flags, 1);

    UPTKError_t ret6 = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret6, UPTKSuccess);
}