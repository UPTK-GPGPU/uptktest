#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 32
#define NUM (WIDTH * WIDTH)
#define SIZE 2048


TEST(cudaDevice, cudaDeviceResetTest)
{
    int DevicesID = 0;
    int numDevices = 0;
    UPTKError_t ret = UPTKSuccess;
    UPTKGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; i++) {
       ret= UPTKSetDevice(i);
       EXPECT_EQ(ret, UPTKSuccess);
       ret = UPTKGetDevice(&DevicesID);
       EXPECT_EQ(i,DevicesID);
       EXPECT_EQ(ret, UPTKSuccess);
    }
    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKGetDevice(&DevicesID);
    EXPECT_EQ(ret, UPTKSuccess);

    //add copy test case
    //sync
    char *buffer = NULL;
    char *hostBuffer = NULL;
    char *deviceBuffer = NULL;
    int bufferSize = 100;

    hostBuffer = (char *)malloc(bufferSize);
    ASSERT_TRUE(hostBuffer!=NULL);
    deviceBuffer = (char *)malloc(bufferSize);
    ASSERT_TRUE(deviceBuffer!=NULL);
  
     for(int i=0; i<bufferSize; i++)
    {
	    hostBuffer[i] = i;
    }

    ret = UPTKMalloc(&buffer, bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(buffer, hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    //  for (int i = 0; i < numDevices; i++) {
    //  ret = UPTKDeviceReset();
    // EXPECT_EQ(ret, UPTKSuccess); 
    // }
    //ret = UPTKDeviceReset();
   // EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKMemcpy(deviceBuffer, buffer, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess);

    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess);

   //add async test case
    UPTKSetDevice(0);
    float *data, *gpuMatrix, *randArray;
    int width = WIDTH;
    randArray = (float*)malloc(NUM * sizeof(float));
    gpuMatrix = (float*)malloc(NUM * sizeof(float));
    for (int i = 0; i < NUM; i++) {
        randArray[i] = (float)i * 1.0f;
    }
    
    UPTKStream_t stream;
    ret = UPTKStreamCreate(&stream);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKMalloc((void**)&data, NUM * sizeof(float));
    UPTKError_t localerror;
    localerror = UPTKMemcpyAsync(data, randArray, NUM * sizeof(float), UPTKMemcpyHostToDevice, stream);
    EXPECT_EQ(localerror, UPTKSuccess);
    localerror = UPTKMemcpyAsync(gpuMatrix, data, NUM * sizeof(float), UPTKMemcpyDeviceToHost, stream); //应该出错
    EXPECT_EQ(localerror, UPTKSuccess);

    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess);

}
