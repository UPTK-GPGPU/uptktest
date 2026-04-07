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
TEST(cudaMemory,cudaMallocTest2){
    int numDevice = 0;
    char **buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKGetDeviceCount(&numDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    std::cout << "numDevice=" << numDevice << std::endl;
    buffer = (char **)malloc(sizeof(char *)*numDevice);
    ASSERT_TRUE(buffer!=NULL);
    memset(buffer, 0, sizeof(char *) * numDevice);
    std::cout << "buffer=" << (void *)buffer << std::endl;
    for (int i = 0; i< numDevice; i++){
        ret = UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);  
        ret = UPTKMalloc(&buffer[i],bufferSize);
        EXPECT_EQ(ret, UPTKSuccess);  
        //memset(buffer[i],0, bufferSize); 
        UPTKPointerAttributes attribs;
        memset(&attribs, 0, sizeof(UPTKPointerAttributes));
        ret = UPTKPointerGetAttributes(&attribs,(void *)buffer[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        EXPECT_EQ(attribs.devicePointer,buffer[i]); 
    }

    for(int i=0; i< numDevice; i++){
        if(buffer[i]){
            ret = UPTKFree(buffer[i]);
            EXPECT_EQ(ret, UPTKSuccess);	
        }
    }

    free(buffer);
}