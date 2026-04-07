#include <stddef.h>
#include <iostream>
#include <stdio.h>
#include <gtest/gtest.h> 
#include <cuda.h>
#include <cuda_runtime_api.h>

#define N 32

TEST(cuP2P,cuMemGetAddressRangeTest)
{
     CUresult ret = CUDA_SUCCESS;
     CUdeviceptr  pbase;
     size_t  psize;
     size_t Nbytes = N * sizeof(float);
     int *A_d;
     cudaSetDevice(0);
    
     cudaMalloc(&A_d, Nbytes);
     ret = cuMemGetAddressRange(&pbase,&psize,(CUdeviceptr)A_d);
     EXPECT_EQ(ret, CUDA_SUCCESS);
     

}