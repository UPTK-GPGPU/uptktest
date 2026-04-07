#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMallocTest6){
    char *buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;

    buffer = (char *)malloc(bufferSize);
    EXPECT_TRUE(buffer!=NULL) << "error: malloc failed";

    // ret = UPTKMalloc(&buffer,bufferSize);
    // EXPECT_EQ(ret, UPTKSuccess);

    // UPTKPointerAttributes attribs;
    // memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    // ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    // EXPECT_EQ(ret, UPTKSuccess);
    // EXPECT_EQ((char *)attribs.devicePointer,buffer);
    
   // free(buffer);

    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKErrorInvalidValue);

    EXPECT_TRUE(buffer!=NULL);

    free(buffer);

}