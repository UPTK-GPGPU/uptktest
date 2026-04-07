#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaDeviceGetSharedMemConfigTest)
{
    UPTKSharedMemConfig pConfig;
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKDeviceGetSharedMemConfig(&pConfig);
    EXPECT_EQ(ret, UPTKSuccess);
}
