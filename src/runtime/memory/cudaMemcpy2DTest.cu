#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMemcpy2DTest){
    UPTKError_t ret = UPTKSuccess;
//#ifndef __TEST_HIPHSA__
    size_t numW = 256;
    size_t numH = 256;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH;
    size_t elements = numW* numH;

    char *A_d;
    char *A_h, *B_h;

    size_t pitch;

    A_h = (char*)malloc(sizeElements);
    EXPECT_TRUE(A_h!=NULL) << "error: malloc failed";
    B_h = (char*)malloc(sizeElements);
    EXPECT_TRUE(B_h!=NULL) << "error: malloc failed";

    for (int i=0; i<elements; i++) {
        A_h[i] = i;
    }

    ret = UPTKMallocPitch((void**)&A_d, &pitch, width, numH);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocPitch failed";

    ret = UPTKMemcpy2D(A_d, pitch, A_h, width, width, numH, UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy2D failed";

    ret = UPTKMemcpy2D(B_h, width, A_d, pitch, width, numH, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy2D failed";

    for (int i=0; i<elements; i++) {
        EXPECT_TRUE (A_h[i] == B_h[i]) << "test case failed, index:" << i << " host value:" << A_h[i] << " device value:" << B_h[i] << std::endl;
    }

    ret = UPTKFree(A_d);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKFree failed"; 
    free(A_h);
    free(B_h);
//#else
    EXPECT_EQ(ret, UPTKSuccess); 
//#endif
}
