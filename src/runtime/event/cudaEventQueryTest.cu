#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 32
#define NUM (WIDTH * WIDTH)
#define SIZE 2048
# define N 32
#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1


__global__ void printf_run_kernel() {  }

TEST(hipEvent, hipEventQueryTest){
     UPTKError_t ret = UPTKSuccess;
     UPTKEvent_t start;
     UPTKEvent_t stop;
     UPTKStream_t stream;
     ret = UPTKStreamCreate(&stream);
     EXPECT_EQ(ret, UPTKSuccess);
     ret = UPTKEventCreate(&start);
     EXPECT_EQ(ret, UPTKSuccess);
     ret = UPTKEventCreate(&stop);
     EXPECT_EQ(ret, UPTKSuccess);
    
    ret = UPTKEventQuery(start);
    std::cout<<ret;
    UPTKEventRecord(start, stream);
    ret = UPTKEventQuery(stop);
    std::cout<<ret;
    //hipLaunchKernelGGL(printf_run_kernel, dim3(1), dim3(1), 0, stream);
    printf_run_kernel<<<dim3(1), dim3(1), 0, stream>>>();
    UPTKEventRecord(stop, stream);
    ret = UPTKEventQuery(start);
    std::cout<<ret;
    ret = UPTKEventQuery(stop);
    std::cout<<ret;
    UPTKEventSynchronize(stop);
    ret = UPTKEventQuery(start);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventQuery(stop);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKEventDestroy(start);
    UPTKEventDestroy(stop);
    ret = UPTKStreamDestroy(stream);
    EXPECT_EQ(ret, UPTKSuccess);
}
