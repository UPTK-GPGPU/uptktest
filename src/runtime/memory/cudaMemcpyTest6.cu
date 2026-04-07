#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaMemory,cudaMemcpyTest6){
    int numDevice = 0;
    float *buffer = NULL;
    float *hostBuffer = NULL;
    float *hostBufferB = NULL;
    int bufferSize = 1024*sizeof(float);
    UPTKError_t ret = UPTKSuccess;

    hostBuffer = (float *)malloc(bufferSize);
    ASSERT_TRUE(hostBuffer!=NULL) << "error: malloc failed";

    hostBufferB = (float *)malloc(bufferSize);
    ASSERT_TRUE(hostBufferB!=NULL) << "error: malloc failed";

    for(int i=0; i<(bufferSize/sizeof(float)); i++)
    {
	    hostBuffer[i] = (i+1)*1.0f;
    }

    ret = UPTKMalloc((void **)&buffer, bufferSize);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc failed"; 
    //free(hostBuffer) << free(hostBufferB);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKHostGetDevicePointer failed";
    EXPECT_EQ(attribs.devicePointer,buffer) << "error: the buffer address is not qual the point";

    //std::cout << "buffer=" << (void *)buffer << std::endl;
    ret = UPTKMemcpy(buffer, hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy failed";

    ret = UPTKMemcpy(hostBufferB, buffer, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy form device to host failed";

    for(int i=0; i<(bufferSize/sizeof(float)); i++){
        EXPECT_TRUE(hostBufferB[i] == hostBuffer[i]) << "index:" << i << " host value:" << hostBuffer[i] << " host B value:" << hostBufferB[i] << std::endl;
        //std::cout << "index:" << i << " host value:" << hostBuffer[i] << " host B value:" << hostBufferB[i] << std::endl;
    }

    free(hostBuffer);
    free(hostBufferB);

    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKSuccess)  << "call UPTKFree failed";	
}