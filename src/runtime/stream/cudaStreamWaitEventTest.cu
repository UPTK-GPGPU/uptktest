#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define NSTREAM 4
#define BDIM 128
using namespace std;
namespace{
void initialData1(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost1(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays1(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    C[idx] = A[idx] + B[idx];

}

bool checkResult1(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            //break;
            return  false;
        }
    }

    if (match) printf("Arrays match.\n\n");
    return  true;
}

TEST(cudaStream, cudaStreamWaitEventTest)
{ // set up data size of vectors
    int nElem = 1 << 18;
    printf("> vector size = %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);
    UPTKError_t ret = UPTKSuccess;

    // malloc pinned host memory for async memcpy
    float *h_A, *h_B, *hostRef, *gpuRef;
    ret = UPTKMallocHost((void**)&h_A, nBytes);
    EXPECT_EQ(ret, UPTKSuccess);   
    ret = UPTKMallocHost((void**)&h_B, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKMallocHost((void**)&gpuRef, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKMallocHost((void**)&hostRef, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 

    // initialize data at host side
    initialData1(h_A, nElem);
    initialData1(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost1(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    ret = UPTKMalloc((float**)&d_A, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKMalloc((float**)&d_B, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKMalloc((float**)&d_C, nBytes);
    EXPECT_EQ(ret, UPTKSuccess); 

    UPTKEvent_t start, stop;
    ret = UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess); 

    // invoke kernel at host side
    dim3 block (BDIM);//BDIM 128
    dim3 grid  ((nElem + block.x - 1) / block.x);

    // grid parallel operation
    int iElem = nElem / NSTREAM;//NSTREAM 4,在NSTREAM流中平均分配该问题的任务
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    UPTKStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        ret = UPTKStreamCreate(&stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
    }

    UPTKEvent_t *kernelEvent;
    kernelEvent = (UPTKEvent_t *) malloc(NSTREAM * sizeof(UPTKEvent_t));

    for (int i = 0; i < NSTREAM; i++)
    {
        ret = UPTKEventCreateWithFlags(&(kernelEvent[i]),
                    UPTK_EVENT_DISABLE_TIMING);
        EXPECT_EQ(ret, UPTKSuccess);
    }

    ret = UPTKEventRecord(start, 0);
    EXPECT_EQ(ret, UPTKSuccess); 

    // initiate all work on the device asynchronously in depth-first order
    for (int i = 0; i < NSTREAM; ++i)
    {
        int ioffset = i * iElem;
        ret = UPTKMemcpyAsync(&d_A[ioffset], &h_A[ioffset], iBytes,UPTKMemcpyHostToDevice, stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        ret = UPTKMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,UPTKMemcpyHostToDevice, stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        ///hipLaunchKernelGGL(sumArrays1, grid, block, 0, stream[i], &d_A[ioffset], &d_B[ioffset],&d_C[ioffset], iElem);
        sumArrays1<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset],&d_C[ioffset], iElem);

        ret = UPTKMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,UPTKMemcpyDeviceToHost, stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 

        ret = UPTKEventRecord(kernelEvent[i], stream[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKStreamWaitEvent(stream[NSTREAM - 1], kernelEvent[i], 0);
        EXPECT_EQ(ret, UPTKSuccess);
    }

    ret = UPTKEventRecord(stop, 0);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventSynchronize(stop);
    EXPECT_EQ(ret, UPTKSuccess); 
    float execution_time;
    ret = UPTKEventElapsedTime(&execution_time, start, stop);
    EXPECT_EQ(ret, UPTKSuccess); 

    printf("\n");
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nBytes * 2e-6) / execution_time );

    // check device results
    //checkResult1(hostRef, gpuRef, nElem);
    EXPECT_EQ(checkResult1(hostRef, gpuRef, nElem), 1);

    // free device global memory
    ret = UPTKFree(d_A);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKFree(d_B);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKFree(d_C);
    EXPECT_EQ(ret, UPTKSuccess); 

    // free host memory
    ret = UPTKFreeHost(h_A);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKFreeHost(h_B);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKFreeHost(hostRef);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKFreeHost(gpuRef);
    EXPECT_EQ(ret, UPTKSuccess); 

    // destroy events
    ret = UPTKEventDestroy(start);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventDestroy(stop);
    EXPECT_EQ(ret, UPTKSuccess); 
    // destroy streams
    for (int i = 0; i < NSTREAM; ++i)
    {
        ret = UPTKStreamDestroy(stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        ret = UPTKEventDestroy(kernelEvent[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }

    free(kernelEvent);
    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess); 
}
}
