#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaEvent, cudaEventCreateWithFlagsTest){
     UPTKError_t ret = UPTKSuccess;
     UPTKEvent_t event;
     unsigned flag = 1;
     ret = UPTKEventCreateWithFlags(&event,flag);
     EXPECT_EQ(ret, UPTKSuccess);

}