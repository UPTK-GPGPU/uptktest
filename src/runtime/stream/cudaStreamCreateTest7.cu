#include <iostream>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define nStreams 32

TEST(cudaStream,cudaStreamCreateTest7)
{
    int numDevices = 0;
    UPTKGetDeviceCount(&numDevices);
//    printf("numDevices = %d\n",numDevices );
    int left_streams = nStreams - numDevices-8;
//    printf("left streams = %d \n",left_streams);
    
    UPTKStream_t stream[nStreams];

    UPTKError_t ret = UPTKSuccess;

    for(int i = 0 ; i < left_streams; i++)
    {
        ret = UPTKStreamCreate(&stream[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }
//    sleep(10);

    for(int i = 0 ; i < left_streams; i++)
    {
        ret = UPTKStreamDestroy(stream[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }
}
