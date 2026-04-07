/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <gtest/gtest.h>
#include <gtest/test_common.h>
#include <cuda.h>

#define N 16
#define offset 3

static __global__ void tex1dKernel(float *val, UPTKTextureObject_t obj) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < (N - offset))
      val[k] = tex1Dfetch<float>(obj, k+offset);
#endif
}


static void runTest(UPTKTextureAddressMode addressMode,
                                         UPTKTextureFilterMode filterMode) {
  UPTKcontext HipContext;
  UPTKdevice HipDevice;
  int deviceID = 0;
  // CUDA_DRIVER_CHECK(cuSetDevice(0));
  CUDA_DRIVER_CHECK(UPDeviceGet(&HipDevice, deviceID));
  CUDA_DRIVER_CHECK(UPCtxCreate(&HipContext, 0, HipDevice));

  // Allocating the required buffer on gpu device
  float *texBuf, *texBufOut;
  float val[N], output[N];

  for (int i = 0; i < N; i++) {
      val[i] = i+1;
      output[i] = 0.0;
  }

  CUDACHECK(UPTKMalloc(&texBuf, N * sizeof(float)));
  CUDACHECK(UPTKMalloc(&texBufOut, N * sizeof(float)));
  CUDACHECK(UPTKMemcpy(texBuf, val, N * sizeof(float), UPTKMemcpyHostToDevice));
  CUDACHECK(UPTKMemset(texBufOut, 0, N * sizeof(float)));
  UPTKResourceDesc resDescLinear;

  memset(&resDescLinear, 0, sizeof(resDescLinear));
  resDescLinear.resType = UPTKResourceTypeLinear;
  resDescLinear.res.linear.devPtr = texBuf;
  resDescLinear.res.linear.desc = UPTKCreateChannelDesc(32, 0, 0, 0,
                                                  UPTKChannelFormatKindFloat);
  resDescLinear.res.linear.sizeInBytes = N * sizeof(float);

  UPTKTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = UPTKReadModeElementType;
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.normalizedCoords = false;

  // Creating texture object
  UPTKTextureObject_t texObj = 0;
  CUDACHECK(UPTKCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

  dim3 dimBlock(1, 1, 1);
  dim3 dimGrid(N, 1, 1);

  tex1dKernel<<<dim3(dimGrid), dim3(dimBlock)>>>(texBufOut, texObj);
  CUDACHECK(UPTKGetLastError()); 
  CUDACHECK(UPTKDeviceSynchronize());

  CUDACHECK(UPTKMemcpy(output, texBufOut, N * sizeof(float),
                                                    UPTKMemcpyDeviceToHost));

  for (int i = 0; i < (N - offset); i++) {
      if (output[i] != val[i + offset]) {
          std::cout<<"Mismatch found at output[" << i << "]:" << output[i] <<
               " val[" << i + offset << "]:" << val[i + offset];
          FAIL();
      }
  }

  for (int i = (N - offset); i < N; i++) {
     if (output[i] != 0) {
          std::cout<<"Output found to be updated at index " << i;
          FAIL();
     }
  }

  CUDACHECK(UPTKDestroyTextureObject(texObj));
  CUDACHECK(UPTKFree(texBuf));
  CUDACHECK(UPTKFree(texBufOut));
}


TEST(cudatexture, cudaTex1DFetchCheckModes) {

  //SECTION("hipAddressModeClamp AND hipFilterModePoint") 
  {
    runTest(UPTKAddressModeClamp, UPTKFilterModePoint);
  }
  //SECTION("hipAddressModeClamp AND hipFilterModeLinear") 
  {
    runTest(UPTKAddressModeClamp, UPTKFilterModeLinear);
  }
  //SECTION("hipAddressModeWrap AND hipFilterModePoint") 
  {
    runTest(UPTKAddressModeWrap, UPTKFilterModePoint);
  }
  //SECTION("hipAddressModeWrap AND hipFilterModeLinear") 
  {
    runTest(UPTKAddressModeWrap, UPTKFilterModeLinear);
  }
}