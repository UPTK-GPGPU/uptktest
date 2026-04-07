#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMallocTest3){
    int numDevice = 0;
    char *buffer = NULL;
    int bufferSize = 0;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMalloc(&buffer,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((uintptr_t)buffer, NULL);
    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);

    EXPECT_EQ(ret, UPTKErrorInvalidValue);

    ret = UPTKFree(buffer);
    //#ifdef __TEST_HIPHSA__
    //EXPECT_EQ(ret, UPTKErrorInvalidValue);
    //#else
    EXPECT_EQ(ret, UPTKSuccess);
    //#endif
}
