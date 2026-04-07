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
#include <iostream>
#include <gtest/gtest.h>
#include <gtest/test_common.h>
typedef float T;

// Texture reference for 2D Layered texture
texture<float, UPTKTextureType2DLayered> tex2DL;

__global__ void simpleKernelLayeredArray(T* outputData,
                                         int width, int height, int layer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[layer * width * height + y * width + x] = tex2DLayered(tex2DL,
                                                              x, y, layer);
}

TEST(cudatexture, cudaSimpleTexture2DLayered) {
  constexpr int SIZE = 512;
  constexpr int num_layers = 5;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int height = SIZE;
  constexpr unsigned int size = width * height * num_layers * sizeof(T);

  T* hData = reinterpret_cast<T*>(malloc(size));
  EXPECT_NE(hData, nullptr);
  memset(hData, 0, size);

  for (unsigned int layer = 0; layer < num_layers; layer++) {
    for (int i = 0; i < static_cast<int>(width * height); i++) {
      hData[layer * width * height + i] = i;
    }
  }
  UPTKChannelFormatDesc channelDesc;
  // Allocate array and copy image data
  channelDesc = UPTKCreateChannelDesc(sizeof(T)*8, 0, 0, 0,
                                     UPTKChannelFormatKindFloat);
  UPTKArray *arr;

  CUDACHECK(UPTKMalloc3DArray(&arr, &channelDesc,
               make_cudaExtent(width, height, num_layers), UPTKArrayLayered));
  UPTKMemcpy3DParms myparms{};
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr = make_cudaPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_cudaExtent(width , height, num_layers);
  // myparms.kind = UPTKMemcpyHostToDevice;
  CUDACHECK(UPTKMemcpy3D(&myparms));

  // set texture parameters
  tex2DL.addressMode[0] = UPTKAddressModeWrap;
  tex2DL.addressMode[1] = UPTKAddressModeWrap;
  tex2DL.filterMode = UPTKFilterModePoint;
  tex2DL.normalized = false;

  // Bind the array to the texture
  CUDACHECK(UPTKBindTextureToArray(tex2DL, arr, channelDesc));

  // Allocate device memory for result
  T* dData = nullptr;
  CUDACHECK(UPTKMalloc(&dData, size));
  EXPECT_NE(dData, nullptr);

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    simpleKernelLayeredArray<<<dimGrid, dimBlock>>>(dData, width, height, layer);
    CUDACHECK(UPTKGetLastError()); 
  }
  CUDACHECK(UPTKDeviceSynchronize());

  // Allocate mem for the result on host side
  T *hOutputData = reinterpret_cast<T*>(malloc(size));
  EXPECT_NE(hOutputData, nullptr);
  memset(hOutputData, 0,  size);

  // copy result from device to host
  CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));
  CudaTest::checkArray(hData, hOutputData, width, height, num_layers);

  CUDACHECK(UPTKFree(dData));
  CUDACHECK(UPTKFreeArray(arr));
  free(hData);
  free(hOutputData);
}