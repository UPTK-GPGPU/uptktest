#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
// 1：调用cudaMalloc申请devce端大小为100字节内存，申请成功
// 2：UPTKPointerGetAttributes()获取地址相关属性，其中devicePoint与申请到的地址相当
// 3：UPTKFree 成功进行内存释放
TEST(cudaMemory,cudaMallocTest){
    int numDevice = 0;
    char *buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret1 = UPTKSuccess;
    UPTKError_t ret2 = UPTKSuccess;
    UPTKError_t ret3 = UPTKSuccess;

    ret1 = UPTKMalloc(&buffer,bufferSize);
    EXPECT_EQ(ret1, UPTKSuccess);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret2 = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret2, UPTKSuccess);
    EXPECT_EQ((char *)attribs.devicePointer,buffer);

    ret3 = UPTKFree(buffer);
    EXPECT_EQ(ret3, UPTKSuccess);
}