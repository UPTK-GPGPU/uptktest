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
void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    C[idx] = A[idx]+B[idx];
    //if (idx < N)
    //{
    //    for (int i = 0; i < N; ++i)
    //    {
    //        C[idx] = A[idx] + B[idx];
    //    }
    //}
}

bool checkResult(float *hostRef, float *gpuRef, const int N)
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
            return false;
        }
    }

    if (match) printf("Arrays match.\n\n");
    return true;
}

TEST(cudaMemory, cudaMemcpyAsyncTest2)
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
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

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
    int iElem = nElem / NSTREAM;//NSTREAM 4,鍦∟STREAM娴佷腑骞冲潎鍒嗛厤璇ラ棶棰樼殑浠诲姟
    size_t iBytes = iElem * sizeof(float);
    grid.x = (iElem + block.x - 1) / block.x;

    UPTKStream_t stream[NSTREAM];

    for (int i = 0; i < NSTREAM; ++i)
    {
        ret = UPTKStreamCreate(&stream[i]);
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
        //UPTKStreamSynchronize(stream[i]);
        ret = UPTKMemcpyAsync(&d_B[ioffset], &h_B[ioffset], iBytes,UPTKMemcpyHostToDevice, stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        //UPTKStreamSynchronize(stream[i]);
        //cudaLaunchKernelGGL(sumArrays, grid, block, 0, stream[i], &d_A[ioffset], &d_B[ioffset],&d_C[ioffset], iElem);

        sumArrays<<<grid, block, 0, stream[i]>>>(&d_A[ioffset], &d_B[ioffset],&d_C[ioffset], iElem);

        //ret = UPTKMemcpyAsync(&gpuRef[ioffset], &d_C[ioffset], iBytes,UPTKMemcpyDeviceToHost, stream[i]);
        //ret = UPTKMemcpyAsync(&gpuRef[ioffset], &d_B[ioffset], iBytes,UPTKMemcpyDeviceToHost, stream[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        //UPTKStreamSynchronize(stream[i]);
    }
    //sleep(10);
    ret = UPTKEventRecord(stop, 0);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventSynchronize(stop);
    EXPECT_EQ(ret, UPTKSuccess); 
    float execution_time;
    ret = UPTKEventElapsedTime(&execution_time, start, stop);
    EXPECT_EQ(ret, UPTKSuccess); 
    
    printf("Actual results from overlapped data transfers:\n");
    printf(" overlap with %d streams : %f ms (%f GB/s)\n", NSTREAM,
           execution_time, (nBytes * 2e-6) / execution_time );

    // check device results
    //checkResult(hostRef, gpuRef, nElem);
    //checkResult(hostRef, d_C, nElem);
    EXPECT_EQ(checkResult(hostRef, d_C, nElem), 1);
    //checkResult(h_B,gpuRef, nElem);

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
    }

    ret = UPTKDeviceReset();
    EXPECT_EQ(ret, UPTKSuccess); 
}
}
