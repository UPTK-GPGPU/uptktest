#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaPointerGetAttributesTest){
    int numDevice = 0;
    char *buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;
    
    ret = UPTKMallocHost(&buffer,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess);  
 
    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess);
    
    EXPECT_EQ(attribs.hostPointer,buffer); 
    EXPECT_EQ(attribs.memoryType,UPTK_MEMORYTYPE_HOST);

    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKSuccess);
}