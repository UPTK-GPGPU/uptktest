#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMemcpy3DTest){
    UPTKError_t ret = UPTKSuccess;
//#ifndef __TEST_HIPHSA__
    int width = 8;
    int height = 8;
    int depth = 8;
    int elements = width * height * depth;
    unsigned int size = width * height * depth * sizeof(char);

    char* hData = (char*) malloc(size);
    memset(hData, 0, size);

    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                hData[i*width*height + j*width +k] = i*width*height + j*width + k;
            }
        }
    }

    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(8, 0, 0, 0, UPTKChannelFormatKindSigned);
    UPTKArray *arr,*arr1;

    ret = UPTKMalloc3DArray(&arr, &channelDesc, make_cudaExtent(width, height, depth), UPTKArrayDefault);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc3DArray failed";
    ret = UPTKMalloc3DArray(&arr1, &channelDesc, make_cudaExtent(width, height, depth), UPTKArrayDefault);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc3DArray failed";

    UPTKMemcpy3DParms myparms = {0};
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcPtr = make_cudaPitchedPtr(hData, width * sizeof(char), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_cudaExtent(width , height, depth);
    myparms.kind = UPTKMemcpyHostToDevice;

    ret = UPTKMemcpy3D(&myparms);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy3D failed";
    UPTKDeviceSynchronize();
    //Array to Array
    memset(&myparms,0x0, sizeof(UPTKMemcpy3DParms));
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
    myparms.extent = make_cudaExtent(width, height, depth);
    myparms.kind = UPTKMemcpyDeviceToDevice;

    ret = UPTKMemcpy3D(&myparms);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy3D failed";
    UPTKDeviceSynchronize();

    char *hOutputData = (char*) malloc(size);
    memset(hOutputData, 0,  size);
    //Device to host
    memset(&myparms,0x0, sizeof(UPTKMemcpy3DParms));
    myparms.srcPos = make_cudaPos(0,0,0);
    myparms.dstPos = make_cudaPos(0,0,0);
    myparms.dstPtr = make_cudaPitchedPtr(hOutputData, width * sizeof(char), width, height);
    myparms.srcArray = arr1;
    myparms.extent = make_cudaExtent(width, height, depth);
    myparms.kind = UPTKMemcpyDeviceToHost;

    ret = UPTKMemcpy3D(&myparms);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy3D failed";
    UPTKDeviceSynchronize();

    // Check result
    for (int i=0; i<elements; i++) {
        EXPECT_TRUE (hData[i] == hOutputData[i]) << "test case failed, index:" << i << " host value:" << hData[i] << " device value:" << hOutputData[i] << std::endl;
    }
    ret = UPTKFreeArray(arr);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKFreeArray failed";
    ret = UPTKFreeArray(arr1);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKFreeArray failed";
    free(hData);
    free(hOutputData);
//#else
    EXPECT_EQ(ret, UPTKSuccess);
//#endif
}
