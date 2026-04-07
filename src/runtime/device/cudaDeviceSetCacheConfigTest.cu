#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice,cudaDeviceSetCacheConfigTest)
{
    UPTKFuncCache cacheConfig = UPTKFuncCachePreferNone;
    UPTKError_t ret = UPTKSuccess;
    ret= UPTKDeviceSetCacheConfig(cacheConfig);
    EXPECT_EQ(ret, UPTKSuccess);
}