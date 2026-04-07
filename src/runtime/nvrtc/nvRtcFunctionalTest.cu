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

/* This test verifies few functional scenarios of hiprtc
 * hipRTC program should be created even if the name passed is empty or null
 * hipRTC program compilation should succeed even if gpu arch is not specified in the options
 * hipRTC should be able to compile kernels using  __forceinline__ keyword
 */
#include <stdio.h>
#include <string.h>
#include <gtest/gtest.h>
#include <gtest/test_common.h>

#include <nvrtc.h>
#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>
const char* funname = "testinline";
static constexpr auto code {
R"(
__forceinline__ __device__ float f() { return 123.4f; }
extern "C"
__global__ void testinline()
{
 f();
}
)"};

TEST(cudanvrtc, nvRtcFunctionalTest) {
  using namespace std;
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));

  nvrtcResult compileResult{nvrtcCompileProgram(prog, 0, 0)};
  size_t logSize;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    cout << log << '\n';
  }
  EXPECT_EQ(compileResult, NVRTC_SUCCESS);
  size_t codeSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &codeSize));

  vector<char> codec(codeSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, codec.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

#if HT_NVIDIA
  int device = 0;
  CUDA_DRIVER_CHECK(UPInit(0));
  UPTKcontext ctx;
  CUDA_DRIVER_CHECK(UPCtxCreate(&ctx, 0, device));
#endif

  UPTKmodule module;
  UPTKfunction function;
  CUDA_DRIVER_CHECK(UPModuleLoadData(&module, codec.data()));
  CUDA_DRIVER_CHECK(UPModuleGetFunction(&function, module, funname));

  CUDA_DRIVER_CHECK(UPLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, nullptr, 0));
  CUDACHECK(UPTKDeviceSynchronize());

  CUDA_DRIVER_CHECK(UPModuleUnload(module));

#if HT_NVIDIA
  CUDA_DRIVER_CHECK(cusCtxDestroy(ctx));
#endif
}
