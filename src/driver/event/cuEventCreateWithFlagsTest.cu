#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuEvent, cuEventCreateTest){
     CUresult ret = CUDA_SUCCESS;
     CUevent event;
     unsigned flag = 1;
     ret = cuEventCreate(&event,flag);
     EXPECT_EQ(ret, CUDA_SUCCESS);

}