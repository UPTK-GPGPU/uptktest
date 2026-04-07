#include <iostream>
#include <gtest/gtest.h>   
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaEvent, cudaEventCreateTest){
    UPTKEvent_t start, stop;
    UPTKError_t ret;
    ret = UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess);
}
