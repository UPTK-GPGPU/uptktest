#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
using namespace std;
namespace{
inline void initMemCpyParam2D(UPTK_MEMCPY2D &ins, const size_t dpitch,
                            const size_t spitch, const size_t width,
                            const size_t height, UPTKmemorytype dstType,
                            UPTKmemorytype srcType) {
  ins.srcXInBytes=0;
  ins.srcY=0;
  ins.srcPitch=spitch;
  ins.dstXInBytes=0;
  ins.dstY=0;
  ins.dstPitch=dpitch;
  ins.WidthInBytes=width;
  ins.Height=height;
  ins.dstMemoryType= dstType;
  ins.srcMemoryType= srcType;
}

TEST(cudaMemory,cudaMemcpyToArrayTest){ 
    size_t numW = 512;
    size_t numH = 1;
    size_t width = numW * sizeof(int);
    size_t sizeElements = width * numH;
    size_t elements = numW* numH;
    
    UPTKArray *A_d;
    int *A_h;
    int *B_h;
    UPTKError_t ret = UPTKSuccess;
#ifdef __TEST_HIPHSA__
    ret = UPTKMallocHost(&A_h, sizeElements);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMallocHost(&B_h, sizeElements);
    EXPECT_EQ(ret, UPTKSuccess);

    for (int i=0; i<elements; i++) {
        A_h[i] = 65;
    }

    UPTKChannelFormatDesc desc = UPTKCreateChannelDesc<float>();
    ret = UPTKMallocArray(&A_d, &desc,numW,numH,0);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocArray failed";

    ret = UPTKMemcpyToArray(A_d, 0, 0, (void*)A_h, width, UPTKMemcpyHostToDevice);//(1维)线性内存
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpyToArray failed";

    UPTK_MEMCPY2D ins;
    initMemCpyParam2D(ins,width,width,width,numH,UPTK_MEMORYTYPE_HOST,UPTK_MEMORYTYPE_ARRAY);
    ins.srcArray    = A_d;
    ins.dstHost     = B_h;
    ret = UPMemcpy2D(&ins);
    EXPECT_EQ(ret, UPTKSuccess);

    for (int i=0; i<elements; i++) {
        EXPECT_TRUE (A_h[i] == B_h[i]) << "test case failed, index:" << i << " host value:" << A_h[i] << " device value:" << B_h[i] << std::endl;
    }
   
    ret = UPTKFreeArray(A_d);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKFreeArray failed";
    UPTKFreeHost(A_h);
    UPTKFreeHost(B_h);
#else
    EXPECT_EQ(ret, UPTKSuccess);
#endif
}
}