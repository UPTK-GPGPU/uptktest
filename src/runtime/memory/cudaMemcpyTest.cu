#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaMemory,cudaMemcpyTest){
    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    char *deviceBuffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;

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

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((char *)attribs.devicePointer,buffer);

    //std::cout << "buffer=" << (void *)buffer << std::endl;
    ret = UPTKMemcpy(buffer, hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(deviceBuffer, buffer, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int i=0; i<bufferSize; i++){
        EXPECT_TRUE(deviceBuffer[i] == hostBuffer[i]);
        //printf("index:%d, host value:%d, device value:%d\n", i, hostBuffer[i], deviceBuffer[i]);
    }

    free(hostBuffer);
    
    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKSuccess);	
}