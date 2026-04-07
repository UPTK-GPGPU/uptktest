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
#include "texture_helper.h"

template<bool normalizedCoords>
__global__ void tex1DKernel(float *outputData, UPTKTextureObject_t textureObject,
                            int width, float offsetX) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[x] = tex1D<float>(textureObject, normalizedCoords ? (x + offsetX) / width : x + offsetX);
#endif
}

template<UPTKTextureAddressMode addressMode, UPTKTextureFilterMode filterMode, bool normalizedCoords>
static void runTest(const int width, const float offsetX) {
  //printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, offsetX=%f)\n", __FUNCTION__,
  //       addressMode, filterMode, normalizedCoords, width, offsetX);
  unsigned int size = width * sizeof(float);
  float *hData = (float*) malloc(size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++) {
    hData[j] = j;
  }

  UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(
      32, 0, 0, 0, UPTKChannelFormatKindFloat);
  UPTKArray *UPTKArray;
  CUDACHECK(UPTKMallocArray(&UPTKArray, &channelDesc, width));

  CUDACHECK(UPTKMemcpy2DToArray(UPTKArray, 0, 0, hData, width * sizeof(float), width * sizeof(float), 1, UPTKMemcpyHostToDevice));

  UPTKResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = UPTKResourceTypeArray;
  resDesc.res.array.array = UPTKArray;

  // Specify texture object parameters
  UPTKTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = UPTKReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  UPTKTextureObject_t textureObject = 0;
  CUDACHECK(UPTKCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

  float *dData = nullptr;
  CUDACHECK(UPTKMalloc((void**) &dData, size));

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid((width + dimBlock.x - 1)/ dimBlock.x, 1, 1);

  tex1DKernel<normalizedCoords><<<dimGrid, dimBlock>>>(dData,textureObject, width, offsetX);
  CUDACHECK(UPTKGetLastError()); 

  CUDACHECK(UPTKDeviceSynchronize());

  float *hOutputData = (float*) malloc(size);
  memset(hOutputData, 0, size);
  CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));

  bool result = true;
  for (int j = 0; j < width; j++) {
    float expectedValue = getExpectedValue<float, addressMode, filterMode>(width, offsetX + j, hData);
    if (!cudaTextureSamplingVerify<float, filterMode>(hOutputData[j], expectedValue)) {
      std::cout<<"Mismatch at " << offsetX + j << ":" << hOutputData[j] <<
           " expected:" << expectedValue << std::endl;
      FAIL();
    }
  }

  CUDACHECK(UPTKDestroyTextureObject(textureObject));
  CUDACHECK(UPTKFree(dData));
  CUDACHECK(UPTKFreeArray(UPTKArray));
  free(hData);
  free(hOutputData);
  //REQUIRE(result);
  ASSERT_TRUE(result);
}

TEST(cudatexture, cudaTextureObj1DCheckModesTest) {

  //SECTION("UPTKAddressModeClamp, UPTKFilterModePoint, regularCoords")
   {
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, false>(256, -3);
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, false>(256, 4);
  }

  //SECTION("UPTKAddressModeBorder, UPTKFilterModePoint, regularCoords")
   {
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, false>(256, -8.5);
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, false>(256, 12.5);
  }

  //SECTION("UPTKAddressModeClamp, UPTKFilterModeLinear, regularCoords")
   {
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, false>(256, -3);
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, false>(256, 4);
  }

  //SECTION("UPTKAddressModeBorder, UPTKFilterModeLinear, regularCoords")
   {
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, false>(256, -8.5);
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, false>(256, 12.5);
  }

  //SECTION("UPTKAddressModeClamp, UPTKFilterModePoint, normalizedCoords")
   {
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, true>(256, -3);
    runTest<UPTKAddressModeClamp, UPTKFilterModePoint, true>(256, 4);
  }

  //SECTION("UPTKAddressModeBorder, UPTKFilterModePoint, normalizedCoords")
   {
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, true>(256, -8.5);
    runTest<UPTKAddressModeBorder, UPTKFilterModePoint, true>(256, 12.5);
  }

  //SECTION("UPTKAddressModeClamp, UPTKFilterModeLinear, normalizedCoords")
   {
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, true>(256, -3);
    runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, true>(256, 4);
  }

  //SECTION("UPTKAddressModeBorder, UPTKFilterModeLinear, normalizedCoords")
   {
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, true>(256, -8.5);
    runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, true>(256, 12.5);
  }
}
