#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define N 100

TEST(cudaMemory,cudaMemestAsyncTest)
{
    int memsetval = 6;
    char *A_d = NULL;
    char *A_h = NULL;
    UPTKStream_t stream;
    size_t buffersize = N*sizeof(char);
    UPTKError_t ret = UPTKSuccess;

    A_h = (char*)malloc(buffersize);
    ret = UPTKMalloc((void**)&A_d, buffersize);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc failed";

    ret = UPTKStreamCreate(&stream);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKStreamCreate failed";

    ret = UPTKMemsetAsync(A_d, memsetval, buffersize, stream);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemsetAsync failed";

    ret = UPTKStreamSynchronize(stream);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKStreamSynchronize failed";

    ret = UPTKMemcpy(A_h, (void*)A_d, buffersize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy failed";

    for(int i=0; i<N; i++){
        EXPECT_TRUE(A_h[i] == memsetval) << "test case failed, index:" << i << " value:" << A_h[i] << "memsetval" << memsetval << std::endl;
    }

    free(A_h);
    ret = UPTKFree((void*)A_d);
    EXPECT_EQ(ret, UPTKSuccess) << "call UPTKFree failed";
    ret = UPTKStreamDestroy(stream) ;
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKStreamDestroy failed";
}