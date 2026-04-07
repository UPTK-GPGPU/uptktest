#include <iostream>
#include <gtest/gtest.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaMemory, cudaMemcpyAsyncTest4) {

    int numDevice = 0;
    float *buffer   = NULL;
    float *hostBuffer = NULL;
    float *hostBufferB = NULL;
    int num = 40000;
    int bufferSize = num*sizeof(float);
    int deviceId = 0;

    UPTKError_t localError;

    localError = UPTKSuccess;

    localError = UPTKGetDevice(&deviceId);
    EXPECT_EQ(localError, UPTKSuccess)<<"get device errpr";

    printf("get device count %d\n", deviceId);
    hostBuffer = (float *)malloc(bufferSize);
    EXPECT_TRUE( hostBuffer != NULL)<<"error: malloc failed\n";
    for(int i=0; i<num;i++)
    {
       hostBuffer[i]=i*3.1415926;
    }
   
    hostBufferB = (float *)malloc(bufferSize);
    EXPECT_TRUE( hostBufferB != NULL)<<"error: malloc failed\n";

    localError = UPTKMalloc((void **)&buffer, bufferSize);
    EXPECT_EQ(localError, UPTKSuccess)<<"UPTKMalloc failed";

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    localError = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(localError, UPTKSuccess)<<"call UPTKHostGetDevicePointer failed";
    EXPECT_EQ(attribs.devicePointer,buffer) << "error: the buffer address is not qual the point";

    localError = UPTKMemcpyAsync(buffer, hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
    EXPECT_EQ(localError, UPTKSuccess)<<"UPTKMemcpy failed";

    localError = UPTKDeviceSynchronize();
    EXPECT_EQ(localError, UPTKSuccess);

    localError = UPTKMemcpy(hostBufferB, buffer, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(localError, UPTKSuccess)<<"UPTKMemcpy form device to host failed";

    for(int i=0; i<num; i++)
    {
        EXPECT_TRUE(hostBufferB[i] == hostBuffer[i]);
    }

    free(hostBuffer);
    free(hostBufferB);
    localError = UPTKFree(buffer);
    EXPECT_EQ(localError, UPTKSuccess)<<"call UPTKFree failed";

    printf("UPTKMemcpyAsync test case success, buffer size %d\n", bufferSize);
}
