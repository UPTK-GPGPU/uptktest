#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
// 1：调用cudaMallocHost申请host端大小为100字pin age内存，申请成功
// 2：UPTKPointerGetAttributes()获取地址相关属性，其中hostPoint与申请到的地址相当
// 3：UPTKFree 成功进行内存释放
TEST(cudaMemory,cudaMallocHostTest){
    int numDevice = 0;
    char *buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKMallocHost(&buffer,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));//把attribs中所有字节换做字符“0”，常用来对指针或字符串的初始化
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    EXPECT_EQ((char *)attribs.hostPointer,buffer);//attribs.devicePointer=0x7fd5f8e00000,buffer=0x7fd5f8e00000
    ret = UPTKFreeHost(buffer);
    EXPECT_EQ(ret, UPTKSuccess);
}