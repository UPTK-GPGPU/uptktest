/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Validates Array Resource texture object with negative/functional tests.
 */
TEST(cudatexture, cudaCreateTextureObject_Array) {

  UPTKError_t ret;
  UPTKResourceDesc resDesc;
  UPTKTextureDesc texDesc;
  UPTKTextureObject_t texObj;

  /* set resource type as hipResourceTypeArray and array(nullptr) */
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = UPTKResourceTypeArray;
  resDesc.res.array.array = nullptr;

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = UPTKReadModeElementType;

  ret = UPTKCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  EXPECT_NE(ret, UPTKSuccess);

/*
 * Validates MipMappedArray Resource texture object
 * with negative/functional tests.
 */
  UPTKError_t ret1;
  UPTKResourceDesc resDesc1;
  UPTKTextureDesc texDesc1;
  UPTKTextureObject_t texObj1;

  /* set resource type as hipResourceTypeMipmappedArray and mipmap(nullptr) */
  // Populate resource descriptor
  memset(&resDesc1, 0, sizeof(resDesc1));
  resDesc1.resType = UPTKResourceTypeMipmappedArray;
  resDesc1.res.mipmap.mipmap = nullptr;

  // Populate texture descriptor
  memset(&texDesc1, 0, sizeof(texDesc1));
  texDesc1.readMode = UPTKReadModeElementType;

  ret1 = UPTKCreateTextureObject(&texObj1, &resDesc1, &texDesc1, nullptr);
  EXPECT_NE(ret1, UPTKSuccess);
}