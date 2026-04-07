#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMallocPitchTest2)
{
    UPTKError_t ret = UPTKSuccess;
    size_t numH = 256;
    size_t numW = 132;
    size_t pitch_A;
    size_t width = numW * sizeof(char);
   
    char *buffer;
    
    ret = UPTKMallocPitch((void**)&buffer, &pitch_A, width , numH);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocPitch failed"; 
    EXPECT_TRUE(pitch_A==256);

    UPTKPointerAttributes attribs;
    memset(&attribs, 0, sizeof(UPTKPointerAttributes));
    ret = UPTKPointerGetAttributes(&attribs,(void *)buffer);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKHostGetDevicePointer failed";
    EXPECT_EQ(attribs.devicePointer,buffer) << "error: the buffer address is not qual the point";

    ret = UPTKFree(buffer);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKFree failed";
    EXPECT_EQ(ret, UPTKSuccess);
}
