#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DriverVersion 11080


TEST(cuDevice,cuDriverGetVersionTest){
    int driverVersion;
    int runtimeVersion;
    CUresult ret1 = CUDA_SUCCESS;
    ret1 = cuDriverGetVersion(&driverVersion);
    //runtimeVersion=3182,driverVersion=318200
    EXPECT_EQ(ret1, CUDA_SUCCESS);
    EXPECT_EQ(driverVersion,DriverVersion);
}