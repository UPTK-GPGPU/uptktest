#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory, cudaMallocArrayTest){
    typedef struct UPTKArray {
    void* data;  // FIXME: generalize this
    struct UPTKChannelFormatDesc desc;
    unsigned int type;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    UPTKarray_format Format;
    unsigned int NumChannels;
    bool isDrv;
    unsigned int textureType;
    }UPTKArray;

    UPTKArray *arr1;
    UPTKError_t ret = UPTKSuccess;
    int width = 8;
    int height = 8;

    UPTKChannelFormatDesc chan_desc=UPTKCreateChannelDesc(32,0,0,0,UPTKChannelFormatKindSigned);
    ret = UPTKMallocArray((UPTKArray_t *)&arr1, &chan_desc,width,height,0);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocArray failed";

    EXPECT_EQ(arr1->width,width);
    EXPECT_EQ(arr1->height,height);

    ret = UPTKFreeArray((UPTKArray_t)arr1);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKFreeArray failed";

}