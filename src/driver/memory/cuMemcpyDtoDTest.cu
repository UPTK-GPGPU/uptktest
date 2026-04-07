#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>


TEST(cuMemory,cuMemcpyDtoDTest){
    int numDevice = 0;
    char *buffer = NULL;
    char *bufferB = NULL;
    char *hostBuffer = NULL;
    char *deviceBuffer = NULL;
    int bufferSize = 100;
    cudaError_t ret = cudaSuccess;
    CUresult ret2 = CUDA_SUCCESS;

    hostBuffer = (char *)malloc(bufferSize);
    EXPECT_TRUE(hostBuffer!=NULL);
    deviceBuffer = (char *)malloc(bufferSize);
    EXPECT_TRUE(deviceBuffer!=NULL);

    for(int i=0; i<bufferSize; i++)
    {
	    hostBuffer[i] = i;
    }

    ret = cudaMalloc(&buffer, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);
    ret = cudaMalloc(&bufferB, bufferSize);
    EXPECT_EQ(ret, cudaSuccess);

    cudaPointerAttributes attribs;
    memset(&attribs, 0, sizeof(cudaPointerAttributes));
    ret = cudaPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ((char *)attribs.devicePointer,buffer);

    ret = cudaMemcpy(buffer, hostBuffer, bufferSize,cudaMemcpyHostToDevice);
    EXPECT_EQ(ret, cudaSuccess);
    ret2 = cuMemcpyDtoD((CUdeviceptr)bufferB, (CUdeviceptr)buffer, bufferSize);
    EXPECT_EQ(ret2, CUDA_SUCCESS);
    ret = cudaMemcpy(deviceBuffer, bufferB, bufferSize,cudaMemcpyDeviceToHost);
    EXPECT_EQ(ret, cudaSuccess);
    for(int i=0; i<bufferSize; i++){
        EXPECT_TRUE(deviceBuffer[i] == hostBuffer[i]);
    }

    free(hostBuffer);
    free(deviceBuffer);
    ret = cudaFree(buffer);
    EXPECT_EQ(ret, cudaSuccess);	
    ret = cudaFree(bufferB);
    EXPECT_EQ(ret, cudaSuccess);
}