#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(UPTKError_t, cudaPeekAtLastErrorTest)
{
    UPTKError_t ret1 = UPTKSuccess;
    UPTKError_t ret2 = UPTKSuccess;
    int deviceId = 2;
    ret1 = UPTKSetDevice(-1);
    EXPECT_EQ(ret1, UPTKErrorInvalidDevice);
    ret2 = UPTKPeekAtLastError();
    EXPECT_NE(ret1, UPTKSuccess);
   
}