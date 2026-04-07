#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory, cudaMalloc3DArrayTest){
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

    UPTKError_t ret = UPTKSuccess;
    UPTKArray * arr1;
    int width = 8;
    int height = 8;
    int depth = 8 ;

    UPTKChannelFormatDesc chan_desc=UPTKCreateChannelDesc(32,0,0,0,UPTKChannelFormatKindSigned);
    ret = UPTKMalloc3DArray((UPTKArray_t *)&arr1, &chan_desc,make_cudaExtent(width,height,depth),UPTKArrayDefault);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocArray failed";
    
    EXPECT_EQ(arr1->width,width);
    EXPECT_EQ(arr1->height,height);
    EXPECT_EQ(arr1->depth,depth);

    ret = UPTKFreeArray((UPTKArray_t)arr1);
    EXPECT_EQ(ret, UPTKSuccess);

}