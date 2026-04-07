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
#include "texture_helper.h"

template<bool normalizedCoords>
__global__ void tex2DKernel(float *outputData, UPTKTextureObject_t textureObject,
                            int width, int height, float offsetX,
                            float offsetY) {
// #if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D<float>(textureObject,
                                           normalizedCoords ? (x + offsetX) / width : x + offsetX,
                                           normalizedCoords ? (y + offsetY) / height : y + offsetY);
// #endif
}

template<UPTKTextureAddressMode addressMode, UPTKTextureFilterMode filterMode, bool normalizedCoords>
static void runTest(const int width, const int height, const float offsetX, const float offsetY) {
  //printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, height=%d, offsetX=%f, offsetY=%f)\n",
  //     __FUNCTION__, addressMode, filterMode, normalizedCoords, width, height, offsetX, offsetY);
  unsigned int size = width * height * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      hData[index] = index;
    }
  }

  UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(
      32, 0, 0, 0, UPTKChannelFormatKindFloat);
  UPTKArray *UPTKArray;
  CUDACHECK(UPTKMallocArray(&UPTKArray, &channelDesc, width, height));

  CUDACHECK(UPTKMemcpy2DToArray(UPTKArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), height, UPTKMemcpyHostToDevice));

  UPTKResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = UPTKResourceTypeArray;
  resDesc.res.array.array = UPTKArray;

  // Specify texture object parameters
  UPTKTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = UPTKReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  UPTKTextureObject_t textureObject = 0;
  CUDACHECK(UPTKCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

  float *dData = nullptr;
  CUDACHECK(UPTKMalloc((void**) &dData, size));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y -1)/ dimBlock.y, 1);

  // cudaLaunchKernelGGL(tex2DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData,
  //                    textureObject, width, height, offsetX, offsetY);
  tex2DKernel<normalizedCoords><<<dimGrid, dimBlock>>>(dData, textureObject, width, height, offsetX, offsetY);  
  CUDACHECK(UPTKGetLastError()); 

  CUDACHECK(UPTKDeviceSynchronize());

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));

  bool result = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      float expectedValue = getExpectedValue<float, addressMode, filterMode>(width, height,
                                                    offsetX + j, offsetY + i, hData);
      if (!cudaTextureSamplingVerify<float, filterMode>(hOutputData[index], expectedValue)) {
        std::cout << "Mismatch at (" << offsetX + j << ", " << offsetY + i << "):" <<
             hOutputData[index] << " expected:" << expectedValue << std::endl;
        result = false;
        goto line1;
      }
    }
  }
line1:
  CUDACHECK(UPTKDestroyTextureObject(textureObject));
  CUDACHECK(UPTKFree(dData));
  CUDACHECK(UPTKFreeArray(UPTKArray));
  free(hData);
  free(hOutputData);
  // REQUIRE(result);
  EXPECT_TRUE(result);
}

TEST(cudatexture, cudaTextureObj2DCheckModes) {

  SECTION("UPTKAddressModeClamp, UPTKFilterModePoint, regularCoords") {
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, false>(256, 256, -3.9, 6.1);
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, false>(256, 256, 4.4, -7.0);
  }

  SECTION("UPTKAddressModeBorder, UPTKFilterModePoint, regularCoords") {
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, false>(256, 256, -8.5, 2.9);
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, false>(256, 256, 12.5, 6.7);
  }

  SECTION("UPTKAddressModeClamp, UPTKFilterModeLinear, regularCoords") {
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, false>(256, 256, -0.4, -0.4);
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, false>(256, 256, 4, 14.6);
  }

  SECTION("UPTKAddressModeBorder, UPTKFilterModeLinear, regularCoords") {
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, false>(256, 256, -0.4, 0.4);
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, false>(256, 256, 12.5, 23.7);
  }

  SECTION("UPTKAddressModeClamp, UPTKFilterModePoint, normalizedCoords") {
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, true>(256, 256, -3, 8.9);
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, true>(256, 256, 4, -0.1);
  }

  SECTION("UPTKAddressModeBorder, UPTKFilterModePoint, normalizedCoords") {
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, true>(256, 256, -8.5, 15.9);
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, true>(256, 256, 12.5, -17.9);
  }

  SECTION("UPTKAddressModeClamp, UPTKFilterModeLinear, normalizedCoords") {
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, true>(256, 256, -3, 5.8);
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, true>(256, 256, 4, 9.1);
  }

  SECTION("UPTKAddressModeBorder, UPTKFilterModeLinear, normalizedCoords") {
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, true>(256, 256, -8.5, 6.6);
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, true>(256, 256, 12.5, 0.01);
  }
}
