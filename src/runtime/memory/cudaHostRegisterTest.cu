#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define LEN 100
using namespace std;
namespace{
__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}


void initialData(float *A, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        A[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
    return;
}

//host端malloc申请内存，调用cudaHostRegister进行锁页，然后将数据copy到device端正常
TEST(cudaMemory,cudaHostRegisterTest){
    float *h_A, *h_B, *hostRef, *gpuRef;
    int bufferSize = LEN*sizeof(float);
    UPTKError_t ret = UPTKSuccess;

    // malloc host memory
    h_A = (float *)malloc(bufferSize);
    EXPECT_TRUE(h_A!=NULL) << "error: malloc failed";

    h_B = (float *)malloc(bufferSize);
    EXPECT_TRUE(h_B!=NULL) << "error: malloc failed";

    hostRef = (float *)malloc(bufferSize);
    EXPECT_TRUE(hostRef!=NULL) << "error: malloc failed";

    gpuRef = (float *)malloc(bufferSize);
    EXPECT_TRUE(gpuRef!=NULL) << "error: malloc failed";

    // allocate zerocpy memory
    ret = UPTKHostRegister(h_A, bufferSize,0); //注册主机内存，以便可以从当前设备访问它
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostRegister failed";
    ret = UPTKHostRegister(h_B, bufferSize,0); //注册主机内存，以便可以从当前设备访问它
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostRegister failed";

    // initialize data at host side
    initialData(h_A, LEN);
    initialData(h_B, LEN);
    memset(hostRef, 0, bufferSize);
    memset(gpuRef,  0, bufferSize);

    // add vector at host side for result checks
    for(int i=0; i<LEN; i++)
    {
	    hostRef[i] = h_A[i] + h_B[i];
    }

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    ret = UPTKMalloc((void **)&d_C,bufferSize);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMalloc failed";

    //pass the pointer to device. 无需在设备上分配内存，也无需在主机内存和设备内存之间拷贝数据。数据传输是在内核需要的时候隐式进行的。
    ret = UPTKHostGetDevicePointer((void **)&d_A, h_A,0);//通过cudaMallocHost分配的主机指针获取设备指针
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostGetDevicePointer failed";
    ret = UPTKHostGetDevicePointer((void **)&d_B, h_B,0);//通过cudaMallocHost分配的主机指针获取设备指针
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKHostGetDevicePointer failed";

    // set up execution configuration
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((LEN + block.x - 1) / block.x);

    // execute kernel with zero copy memory
    //hipLaunchKernelGGL(sumArraysZeroCopy, grid, block, 0, 0, d_A, d_B, d_C, LEN);
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, LEN);

    UPTKDeviceSynchronize();

    // copy kernel result back to host side
    ret = UPTKMemcpy(gpuRef, d_C, bufferSize, UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess) << "UPTKMemcpy form device to host failed";

    for(int i=0; i<LEN; i++){
        EXPECT_TRUE(hostRef[i] == gpuRef[i]) << "test case failed, index:" << i << " host value:" << hostRef[i] << " devicePoint value:" << gpuRef[i] << std::endl;
    }

    UPTKHostUnregister(h_A);
    UPTKHostUnregister(h_B);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    ret = UPTKFree(d_C);
    EXPECT_EQ(ret, UPTKSuccess)  << "call UPTKFree failed";

}
}