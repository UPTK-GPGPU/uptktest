#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaMemory,cudaMemset2DTest)
{
    UPTKError_t ret = UPTKSuccess;
//#ifndef __TEST_HIPHSA__
    size_t numH = 2;
    size_t numW = 2;
    size_t pitch_A;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH;
    size_t elements = numW* numH;
   
    int memsetval = 5; 
    char *A_d;
    char *A_h;
    
    //向设备分配至少widthInBytes*height字节的线性内存
    //并以*A_d的形式返回指向所分配存储器的指针,函数将确保在任何给出的行中对应的指针是连续的
    //以*pitch的形式返回分配的宽度，以字节为单位.用来计算2D数组中的地址
    ret = UPTKMallocPitch((void**)&A_d, &pitch_A, width , numH);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMallocPitch failed"; 

    A_h = (char*)malloc(sizeElements);
    EXPECT_TRUE(A_h != NULL);
    for (size_t i=0; i<elements; i++) {
        A_h[i] = 1;
    }
    ret = UPTKMemset2D(A_d, pitch_A, memsetval, numW, numH);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemset2D failed"; 
    //width是A_h指向的2D数组中的内存宽度
    ret = UPTKMemcpy2D(A_h, width, A_d, pitch_A, numW, numH, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy2D failed"; 

    for (int i=0; i<elements; i++) {
        EXPECT_TRUE (A_h[i] == memsetval) << "test case failed, index:" << i << " value:" << A_h[i] << "memsetval" << memsetval << std::endl;
    }

    ret = UPTKFree(A_d);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKFree failed"; 
    free(A_h);
//#else
     EXPECT_EQ(ret, UPTKSuccess);
//#endif
}
