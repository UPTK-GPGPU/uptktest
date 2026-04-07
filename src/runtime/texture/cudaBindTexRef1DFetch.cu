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
#include <gtest/gtest.h>
#include <gtest/test_common.h>

#define N 512

texture<float, 1, UPTKReadModeElementType> tex;

static __global__ void kernel(float *out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
      out[x] = tex1Dfetch(tex, x);
  }
}

TEST(cudatexture, cudaBindTexRef1DFetch) {

  float *texBuf;
  float val[N], output[N];
  size_t offset = 0;
  float *devBuf;
  for (int i = 0; i < N; i++) {
      val[i] = i;
      output[i] = 0.0;
  }
  UPTKChannelFormatDesc chanDesc =
      UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);

  CUDACHECK(UPTKMalloc(&texBuf, N * sizeof(float)));
  CUDACHECK(UPTKMalloc(&devBuf, N * sizeof(float)));
  CUDACHECK(UPTKMemcpy(texBuf, val, N * sizeof(float), UPTKMemcpyHostToDevice));

  tex.addressMode[0] = UPTKAddressModeClamp;
  tex.addressMode[1] = UPTKAddressModeClamp;
  tex.filterMode = UPTKFilterModePoint;
  tex.normalized = 0;

  CUDACHECK(UPTKBindTexture(&offset, tex, reinterpret_cast<void *>(texBuf),
                                               chanDesc, N * sizeof(float)));
  CUDACHECK(UPTKGetTextureAlignmentOffset(&offset, &tex));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  kernel<<<dim3(dimGrid), dim3(dimBlock)>>>(devBuf);
  CUDACHECK(UPTKGetLastError()); 
  CUDACHECK(UPTKDeviceSynchronize());
  CUDACHECK(UPTKMemcpy(output, devBuf, N * sizeof(float), UPTKMemcpyDeviceToHost));
  int errors = 0;
  for (int i = 0; i < N; i++) {
      if (output[i] != val[i]) {
        errors++;
      }
  }
  EXPECT_EQ(errors, 0);

  CUDACHECK(UPTKUnbindTexture(&tex));
  CUDACHECK(UPTKFree(texBuf));
  CUDACHECK(UPTKFree(devBuf));
}
