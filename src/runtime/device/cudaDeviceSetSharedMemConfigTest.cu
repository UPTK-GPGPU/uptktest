#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice, cudaDeviceSetSharedMemConfigTest)
{
    UPTKSharedMemConfig pConfig = UPTKSharedMemBankSizeDefault;
    UPTKError_t ret = UPTKSuccess;
    ret= UPTKDeviceSetSharedMemConfig(pConfig);
    EXPECT_EQ(ret, UPTKSuccess);
}
