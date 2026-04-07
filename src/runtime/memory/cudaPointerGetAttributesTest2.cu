#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaPointerGetAttributesTest2){
    int numDevice = 0;
    char **buffer = NULL;
    int bufferSize = 100;
    UPTKError_t ret = UPTKSuccess;

    ret = UPTKGetDeviceCount(&numDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    
    buffer = (char **)malloc(sizeof(char *)*numDevice);
    ASSERT_TRUE(buffer!=NULL);
    memset(buffer, 0, sizeof(char *) * numDevice);
    
    for (int i = 0; i< numDevice; i++){
        ret = UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);  

        ret = UPTKMalloc(&buffer[i],bufferSize);
        EXPECT_EQ(ret, UPTKSuccess);  
 
        UPTKPointerAttributes attribs;
        memset(&attribs, 0, sizeof(UPTKPointerAttributes));
        ret = UPTKPointerGetAttributes(&attribs,(void *)buffer[i]);
 
        EXPECT_EQ(ret, UPTKSuccess);
        EXPECT_EQ(attribs.devicePointer,buffer[i]); 

        EXPECT_EQ(attribs.memoryType,UPTK_MEMORYTYPE_DEVICE);
        EXPECT_EQ(attribs.device,i); 
    }

    for(int i=0; i< numDevice; i++){
        if(buffer[i]){
            ret = UPTKFree(buffer[i]);
            EXPECT_EQ(ret, UPTKSuccess);	
        }
    }

    free(buffer);
}
    

