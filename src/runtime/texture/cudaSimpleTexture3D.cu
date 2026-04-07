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

// Texture reference for 3D texture
texture<float, UPTKTextureType3D, UPTKReadModeElementType> texf;
texture<int, UPTKTextureType3D, UPTKReadModeElementType>   texi;
texture<char, UPTKTextureType3D, UPTKReadModeElementType>  texc;

template <typename T>
__global__ void simpleKernel3DArray(T* outputData, int width,
                                    int height, int depth) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        if (std::is_same<T, float>::value)
          outputData[i*width*height + j*width + k] = tex3D(texf, k, j, i);
        else if (std::is_same<T, int>::value)
          outputData[i*width*height + j*width + k] = tex3D(texi, k, j, i);
        else if (std::is_same<T, char>::value)
          outputData[i*width*height + j*width + k] = tex3D(texc, k, j, i);
      }
    }
  }
#endif
}

template <typename T>
static void runSimpleTexture3D_Check(int width, int height, int depth,
            texture<T, UPTKTextureType3D, UPTKReadModeElementType> *tex) {
  unsigned int size = width * height * depth * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  EXPECT_NE(hData, nullptr);
  memset(hData, 0, size);

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        hData[i*width*height + j*width +k] = i*width*height + j*width + k;
      }
    }
  }

  // Allocate array and copy image data
  UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc<T>();
  UPTKArray *arr;

  CUDACHECK(UPTKMalloc3DArray(&arr, &channelDesc,
            make_cudaExtent(width, height, depth), UPTKArrayDefault));
  UPTKMemcpy3DParms myparms{};
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr = make_cudaPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = arr;
  myparms.extent = make_cudaExtent(width, height, depth);
  myparms.kind = UPTKMemcpyHostToDevice;

  CUDACHECK(UPTKMemcpy3D(&myparms));

  // set texture parameters
  tex->addressMode[0] = UPTKAddressModeWrap;
  tex->addressMode[1] = UPTKAddressModeWrap;
  tex->filterMode = UPTKFilterModePoint;
  tex->normalized = false;

  // Bind the array to the texture
  CUDACHECK(UPTKBindTextureToArray(*tex, arr, channelDesc));

  // Allocate device memory for result
  T* dData = nullptr;
  CUDACHECK(UPTKMalloc(&dData, size));
  EXPECT_NE(dData, nullptr);

  simpleKernel3DArray<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(dData, width, height, depth);
  CUDACHECK(UPTKGetLastError()); 
  CUDACHECK(UPTKDeviceSynchronize());

  // Allocate mem for the result on host side
  T *hOutputData = reinterpret_cast<T*>(malloc(size));
  EXPECT_NE(hOutputData, nullptr);
  memset(hOutputData, 0,  size);

  // copy result from device to host
  CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));
  CudaTest::checkArray(hData, hOutputData, width, height, depth);

  CUDACHECK(UPTKFree(dData));
  CUDACHECK(UPTKFreeArray(arr));
  free(hData);
  free(hOutputData);
}

TEST(cudatexture, cudaSimpleTexture3D) {
  for ( int i = 1; i < 25; i++ ) {
    runSimpleTexture3D_Check<float>(i, i, i, &texf);
    runSimpleTexture3D_Check<int>(i+1, i, i, &texi);
    runSimpleTexture3D_Check<char>(i, i+1, i, &texc);
  }
}