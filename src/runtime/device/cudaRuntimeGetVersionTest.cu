#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#ifdef __TEST_HIPHSA__
#define RuntimeVersion 20494
#else
#define RuntimeVersion 3182
#endif


TEST(cudaDevice,cudaRuntimeGetVersionTest){
    int runtimeVersion;
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKRuntimeGetVersion(&runtimeVersion);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_NE(runtimeVersion, 0);
}
