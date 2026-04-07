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

#define SIZE_H 20
#define SIZE_W 179

// texture object is a kernel argument
template <typename TYPE_t>
static __global__ void texture2dCopyKernel(UPTKTextureObject_t texObj,
                                                   TYPE_t* dst) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  for (int i = 0; i < SIZE_H; i++)
      for (int j = 0; j < SIZE_W; j++)
          dst[SIZE_W*i+j] = tex2D<TYPE_t>(texObj, j, i);
  __syncthreads();
#endif
}


TEST(cudatexture, cudaTexObjPitch) {

  float* B;
  float* A;
  float* devPtrB;
  float* devPtrA;

  B = new float[SIZE_H*SIZE_W];
  A = new float[SIZE_H*SIZE_W];
  for (size_t i=1; i <= (SIZE_H*SIZE_W); i++) {
      A[i-1] = i;
  }

  size_t devPitchA;
  CUDACHECK(UPTKMallocPitch(reinterpret_cast<void**>(&devPtrA), &devPitchA,
                                       SIZE_W*sizeof(float), SIZE_H));
  CUDACHECK(UPTKMemcpy2D(devPtrA, devPitchA, A, SIZE_W*sizeof(float),
          SIZE_W*sizeof(float), SIZE_H, UPTKMemcpyHostToDevice));

  // Use the texture object
  UPTKResourceDesc texRes;
  memset(&texRes, 0, sizeof(texRes));
  texRes.resType = UPTKResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = devPtrA;
  texRes.res.pitch2D.height = SIZE_H;
  texRes.res.pitch2D.width = SIZE_W;
  texRes.res.pitch2D.pitchInBytes = devPitchA;
  texRes.res.pitch2D.desc = UPTKCreateChannelDesc<float>();

  UPTKTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(texDescr));
  texDescr.normalizedCoords = false;
  texDescr.filterMode = UPTKFilterModePoint;
  texDescr.mipmapFilterMode = UPTKFilterModePoint;
  texDescr.addressMode[0] = UPTKAddressModeClamp;
  texDescr.addressMode[1] = UPTKAddressModeClamp;
  texDescr.addressMode[2] = UPTKAddressModeClamp;
  texDescr.readMode = UPTKReadModeElementType;

  UPTKTextureObject_t texObj;
  CUDACHECK(UPTKCreateTextureObject(&texObj, &texRes, &texDescr, NULL));

  CUDACHECK(UPTKMalloc(reinterpret_cast<void**>(&devPtrB),
                                     SIZE_W*sizeof(float)*SIZE_H));

  texture2dCopyKernel<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(texObj, devPtrB);
  CUDACHECK(UPTKGetLastError()); 

  CUDACHECK(UPTKMemcpy2D(B, SIZE_W*sizeof(float), devPtrB,
                        SIZE_W*sizeof(float), SIZE_W*sizeof(float),
                                         SIZE_H, UPTKMemcpyDeviceToHost));

  CudaTest::checkArray(A, B, SIZE_H, SIZE_W);
  delete []A;
  delete []B;
  CUDACHECK(UPTKFree(devPtrA));
  CUDACHECK(UPTKFree(devPtrB));
}