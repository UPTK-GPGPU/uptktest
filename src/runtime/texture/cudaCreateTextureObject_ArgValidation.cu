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
#define N 512

/*
 * Validate argument list of texture object api.
 */
TEST(cudatexture, cudaCreateTextureObject_ArgValidation) {

  float *texBuf;
  UPTKError_t ret;
  constexpr int xsize = 32;
  UPTKResourceDesc resDesc;
  UPTKTextureDesc texDesc;
  UPTKTextureObject_t texObj;

  /** Initialization */
  CUDACHECK(UPTKMalloc(&texBuf, N * sizeof(float)));
  // Populate resource descriptor
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = UPTKResourceTypeLinear;
  resDesc.res.linear.devPtr = texBuf;
  resDesc.res.linear.desc = UPTKCreateChannelDesc(xsize, 0, 0, 0,
                       UPTKChannelFormatKindFloat);
  resDesc.res.linear.sizeInBytes = N * sizeof(float);

  // Populate texture descriptor
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = UPTKReadModeElementType;


  /** Sections */
  //SECTION("TextureObject as nullptr") 
  {
    ret = UPTKCreateTextureObject(nullptr, &resDesc, &texDesc, nullptr);
    EXPECT_NE(ret, UPTKSuccess);
  }

  //SECTION("Resouce Descriptor as nullptr") 
  {
    // ret = UPTKCreateTextureObject(&texObj, nullptr, &texDesc, nullptr);
    // EXPECT_NE(ret, UPTKSuccess);
  }

  //SECTION("Texture Descriptor as nullptr") 
  {
    //if ((TestContext::get()).isAmd()) {
      // ret = UPTKCreateTextureObject(&texObj, &resDesc, nullptr, nullptr);
      // EXPECT_NE(ret, UPTKSuccess);
    //} else {
      // API expected to return failure. Test skipped
      // on nvidia as api returns success and would lead
      // to unexpected behavior with app.
      //WARN("Texture Desc(nullptr) skipped on nvidia");
    //}
  }

  //SECTION("Destroy TextureObject with nullptr") 
  {
    ret = UPTKDestroyTextureObject((UPTKTextureObject_t)nullptr);
    // api to return success and no crash seen.
    EXPECT_EQ(ret, UPTKSuccess);
  }

  /** De-Initialization */
  CUDACHECK(UPTKFree(texBuf));
}