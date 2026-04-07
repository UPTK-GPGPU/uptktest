#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


#define LEN 1024  

void initialData3(char *A, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        A[i] = (char)( rand() & 0xFF ) / 10.0f;
    }
    return;
}

//host端malloc申请内存，调用cudaHostRegister进行锁页，然后将数据copy到device端正常
TEST(cudaMemory,cudaHostRegisterTest3){
    char *h_A, *h_B;
    char *d_A;
    unsigned bufferSize = LEN*sizeof(char);
    UPTKError_t ret = UPTKSuccess;

    // malloc host memory
    h_A = (char *)malloc(bufferSize);
    EXPECT_TRUE(h_A!=NULL) << "error: malloc failed";

    h_B = (char *)malloc(bufferSize);
    EXPECT_TRUE(h_B!=NULL) << "error: malloc failed";

    // allocate zerocpy memory
    ret = UPTKHostRegister(h_A, bufferSize,0); //注册主机内存，以便可以从当前设备访问它
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostRegister failed";

    // initialize data at host side
    initialData3(h_A, LEN);
    memset(h_B, 0, bufferSize);    

    ret = UPTKMalloc((void **)&d_A,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc failed";

    UPTKMemcpyAsync(d_A, h_A, bufferSize, UPTKMemcpyHostToDevice, NULL);
    
    UPTKDeviceSynchronize();

    // copy kernel result back to host side
    ret = UPTKMemcpy(h_B, d_A, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy form device to host failed";

    for(int i=0; i<LEN; i++){
        EXPECT_TRUE(h_A[i] == h_B[i]);
    }

    free(h_A);
    free(h_B);

    ret = UPTKHostUnregister(h_A);
    EXPECT_EQ(ret, UPTKSuccess);

    ret = UPTKFree(d_A);
    EXPECT_EQ(ret, UPTKSuccess)  << "call UPTKFree failed";

}