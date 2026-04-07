#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>


TEST(cuMemory,cuMemcpyDtoDAsyncTest2){
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

    cudaStream_t stream;
    ret = cudaStreamCreate(&stream);
    EXPECT_EQ(ret, cudaSuccess);
    ret = cudaMemcpyAsync(buffer, hostBuffer, bufferSize,cudaMemcpyHostToDevice, stream);
    EXPECT_EQ(ret, cudaSuccess);
    ret2 = cuMemcpyDtoDAsync((CUdeviceptr)bufferB, (CUdeviceptr)buffer, bufferSize, NULL);
    EXPECT_EQ(ret2, CUDA_SUCCESS);
    ret = cudaMemcpyAsync(deviceBuffer, bufferB, bufferSize,cudaMemcpyDeviceToHost, stream);
    EXPECT_EQ(ret, cudaSuccess);

    cudaDeviceSynchronize();
    for(int i=0; i<bufferSize; i++){
        EXPECT_TRUE(deviceBuffer[i] == hostBuffer[i]);
    }

    free(hostBuffer);
    free(deviceBuffer);
    ret = cudaFree(buffer);
    EXPECT_EQ(ret, cudaSuccess);	
    ret = cudaFree(bufferB);
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaStreamDestroy(stream);
    EXPECT_EQ(ret, cudaSuccess); 
}
