#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMallocHostTest2){
    char *buffer = NULL;
    int bufferSize = 1073741824;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMallocHost(&buffer,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((char *)attribs.hostPointer,buffer);//attribs.devicePointer=0x7fd5f8e00000,buffer=0x7fd5f8e00000

    
    for(int i=0; i<bufferSize; i++){
	    buffer[i] = 65;
    }

    for(int i=0; i<bufferSize; i++){
	    EXPECT_TRUE(buffer[i] == 65);
    }

    ret = UPTKFreeHost(buffer);
    EXPECT_EQ(ret, UPTKSuccess);

    ret = UPTKFreeHost(buffer);
    EXPECT_EQ(ret, UPTKErrorInvalidValue);
}