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

#define SIZE          10
#define EPSILON       0.00001
#define THRESH_HOLD   0.01  // For filter mode

static float getNormalizedValue(const float value,
                                const UPTKChannelFormatDesc& desc) {
  if ((desc.x == 8) && (desc.f == UPTKChannelFormatKindSigned))
    return (value / SCHAR_MAX);
  if ((desc.x == 8) && (desc.f == UPTKChannelFormatKindUnsigned))
    return (value / UCHAR_MAX);
  if ((desc.x == 16) && (desc.f == UPTKChannelFormatKindSigned))
    return (value / SHRT_MAX);
  if ((desc.x == 16) && (desc.f == UPTKChannelFormatKindUnsigned))
    return (value / USHRT_MAX);
  return value;
}

texture<char, UPTKTextureType1D, UPTKReadModeNormalizedFloat>            texc;
texture<unsigned char, UPTKTextureType1D, UPTKReadModeNormalizedFloat>   texuc;

template<typename T>
__global__ void normalizedValTextureTest(unsigned int numElements,
                                         float* pDst) {
// #if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  unsigned int elementID = threadIdx.x;
  if (elementID >= numElements)
    return;
  float coord = elementID/static_cast<float>(numElements);
  if (std::is_same<T, char>::value)
    pDst[elementID] = tex1D(texc, coord);
  else if (std::is_same<T, unsigned char>::value)
    pDst[elementID] = tex1D(texuc, coord);
// #endif
}

static void textureVerifyFilterModePoint(float *hOutputData,
                                         float *expected, int size) {
  for (int i = 0; i < size; i++) {
    if ((hOutputData[i] == expected[i])
        || (i >= 1 && hOutputData[i] == expected[i - 1]) ||  // round down
        (i < (size - 1) && hOutputData[i] == expected[i + 1])) {  // round up
      continue;
    }
      std::cout<<"Mismatch at output[" << i << "]:" << hOutputData[i] <<
           " expected[" << i << "]:" << expected[i];
    if (i >= 1) {
      std::cout<<", expected[" << i - 1 << "]:" << expected[i - 1];
    }
    if (i < (size - 1)) {
      std::cout<<", expected[" << i + 1 << "]:" << expected[i + 1];
    }
   FAIL();
  }
}

static void textureVerifyFilterModeLinear(float *hOutputData,
                                          float *expected,  int size) {
  for (int i = 0; i < size; i++) {
    float mean = (fabs(expected[i]) + fabs(hOutputData[i])) / 2;
    float ratio = fabs(expected[i] - hOutputData[i]) / (mean + EPSILON);
    if (ratio > THRESH_HOLD) {
      std::cout<<"Mismatch found at output[" << i << "]:" << hOutputData[i] <<
           " expected[" << i << "]:" << expected[i] << ", ratio:" << ratio;
      FAIL();
    }
  }
}

template<UPTKTextureFilterMode fMode = UPTKFilterModePoint>
static void textureVerify(float *hOutputData, float *expected, size_t size) {
  if (fMode == UPTKFilterModePoint) {
    textureVerifyFilterModePoint(hOutputData, expected, size);
  } else if (fMode == UPTKFilterModeLinear) {
    textureVerifyFilterModeLinear(hOutputData, expected, size);
  }
}

template<typename T, UPTKTextureFilterMode fMode = UPTKFilterModePoint>
static void textureTest(texture<T, UPTKTextureType1D,
                        UPTKReadModeNormalizedFloat> *tex) {
  UPTKChannelFormatDesc desc = UPTKCreateChannelDesc<T>();
  UPTKArray_t dData;
  CUDACHECK(UPTKMallocArray(&dData, &desc, SIZE, 1, UPTKArrayDefault));

  T hData[] = {65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
  CUDACHECK(UPTKMemcpy2DToArray(dData, 0, 0, hData, sizeof(T) * SIZE,
            sizeof(T) * SIZE, 1, UPTKMemcpyHostToDevice));

  tex->normalized = true;
  tex->channelDesc = desc;
  tex->filterMode = fMode;
  CUDACHECK(UPTKBindTextureToArray(tex, dData, &desc));

  float *dOutputData = NULL;
  CUDACHECK(UPTKMalloc(&dOutputData, sizeof(float) * SIZE));
  EXPECT_NE(dOutputData, nullptr);

  normalizedValTextureTest<T><<<dim3(1, 1, 1), dim3(SIZE, 1, 1)>>>(SIZE, dOutputData);
  CUDACHECK(UPTKGetLastError()); 

  float *hOutputData = new float[SIZE];
  EXPECT_NE(hOutputData, nullptr);
  CUDACHECK(UPTKMemcpy(hOutputData, dOutputData, (sizeof(float) * SIZE),
                     UPTKMemcpyDeviceToHost));

  float expected[SIZE];
  for (int i = 0; i < SIZE; i++) {
    expected[i] = getNormalizedValue(static_cast<float>(hData[i]), desc);
  }
  textureVerify<fMode>(hOutputData, expected, SIZE);

  CUDACHECK(UPTKFreeArray(dData));
  CUDACHECK(UPTKFree(dOutputData));
  delete [] hOutputData;
}

template<UPTKTextureFilterMode fMode = UPTKFilterModePoint>
static void runTest_cudaTextureFilterMode() {
  textureTest<char, fMode>(&texc);
  textureTest<unsigned char, fMode>(&texuc);
}

TEST(cudatexture, cudaNormalizedFloatValueTex) {

// #if HT_AMD
  UPTKDeviceProp props;
  CUDACHECK(UPTKGetDeviceProperties(&props, 0));
  std::cout<<"Device :: " << props.name << std::endl;
  // std::cout<<"Arch - AMD GPU :: " << props.gcnArch);
// #endif

  //SECTION("hipNormalizedFloatValueTexture for hipFilterModePoint") {
    runTest_cudaTextureFilterMode<UPTKFilterModePoint>();
  //}
  //SECTION("hipNormalizedFloatValueTexture for hipFilterModeLinear") {
    runTest_cudaTextureFilterMode<UPTKFilterModeLinear>();
  //}
}