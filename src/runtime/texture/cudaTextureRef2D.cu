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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <string.h>
#include <gtest/gtest.h>
#include <gtest/test_common.h>
#include <iostream>

texture<float, 2, UPTKReadModeElementType> tex;

__global__ void tex2DKernel(float* outputData, int width) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D(tex, x, y);
#endif
}

TEST(cudatexture, cudaTextureRef2D) {

  constexpr int SIZE = 256;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int height = SIZE;
  constexpr unsigned int size = width * height * sizeof(float);
  unsigned int i, j;

  float* hData = reinterpret_cast<float*>(malloc(size));
  EXPECT_NE(hData, nullptr);
  memset(hData, 0, size);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      hData[i * width + j] = i * width + j;
    }
  }

  UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0,
                                           UPTKChannelFormatKindFloat);
  UPTKArray* UPTKArray;
  CUDACHECK(UPTKMallocArray(&UPTKArray, &channelDesc, width, height));
  CUDACHECK(UPTKMemcpyToArray(UPTKArray, 0, 0, hData, size,
                             UPTKMemcpyHostToDevice));

  tex.addressMode[0] = UPTKAddressModeWrap;
  tex.addressMode[1] = UPTKAddressModeWrap;
  tex.filterMode = UPTKFilterModePoint;
  tex.normalized = 0;

  CUDACHECK(UPTKBindTextureToArray(tex, UPTKArray, channelDesc));

  float* dData = nullptr;
  CUDACHECK(UPTKMalloc(&dData, size));
  EXPECT_NE(dData, nullptr);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  tex2DKernel<<<dim3(dimGrid), dim3(dimBlock)>>>(dData, width);
  CUDACHECK(UPTKGetLastError()); 
  CUDACHECK(UPTKDeviceSynchronize());

  float* hOutputData = reinterpret_cast<float*>(malloc(size));
  EXPECT_NE(hOutputData, nullptr);
  memset(hOutputData, 0, size);
  CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        FAIL()<<"Difference found at [ " << i << j << " ]: " <<
              hData[i * width + j] << hOutputData[i * width + j];
      }
    }
  }
  CUDACHECK(UPTKUnbindTexture(tex));
  CUDACHECK(UPTKFree(dData));
  CUDACHECK(UPTKFreeArray(UPTKArray));
}
