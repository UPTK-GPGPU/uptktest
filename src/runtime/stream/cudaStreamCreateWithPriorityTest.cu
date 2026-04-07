#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaStream,cudaStreamCreateWithPriorityTest){
    int priority_low, priority_high,priority_normal,priority;//1,1,-1
    bool enable_priority_normal = false;
    UPTKStream_t stream;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    EXPECT_EQ(ret, UPTKSuccess);

    EXPECT_TRUE(priority_low - priority_high !=0);

    if ((priority_low - priority_high) > 1) enable_priority_normal = true;
    if (enable_priority_normal) priority_normal = (priority_low - priority_high) / 2;

    ret = UPTKStreamCreateWithPriority(&stream, UPTKStreamDefault, priority_normal);
    EXPECT_EQ(ret, UPTKSuccess);

    ret = UPTKStreamGetPriority(stream, &priority);
    EXPECT_EQ(ret, UPTKSuccess);

    EXPECT_TRUE(priority_normal == priority);

    ret = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret, UPTKSuccess); 
}
