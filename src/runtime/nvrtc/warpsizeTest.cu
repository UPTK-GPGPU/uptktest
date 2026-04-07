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

#include <nvrtc.h>
#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

const char* funcname = "getWarpSize";
static constexpr auto code{
    R"(
extern "C"
__global__
void getWarpSize(int* warpSizePtr)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) *warpSizePtr = warpSize;
}
)"};

TEST(cudanvrtc, warpsizeTest) {
  using namespace std;
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, code, "code.cu", 0, nullptr, nullptr));

  UPTKDeviceProp props;
  int device = 0;
  CUDACHECK(UPTKGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props->gcnArchName;
#else
  std::string sarg = std::string("--gpu-architecture=compute_")
    + std::to_string(props.major) + std::to_string(props.minor);
#endif
  vector<const char*> opts;
  opts.push_back(sarg.c_str());

  nvrtcResult compileResult{nvrtcCompileProgram(prog, opts.size(), opts.data())};
  size_t logSize;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  EXPECT_EQ(compileResult, NVRTC_SUCCESS);
  size_t codeSize;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &codeSize));

  vector<char> codec(codeSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, codec.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  int* d_warpSize;
  CUDA_DRIVER_CHECK(UPMemAlloc((UPTKdeviceptr *)&d_warpSize, sizeof(int)));

  UPTKmodule module;
  UPTKfunction function;
  CUDA_DRIVER_CHECK(UPModuleLoadData(&module, codec.data()));
  CUDA_DRIVER_CHECK(UPModuleGetFunction(&function, module, funcname));

  void* args[] = { &d_warpSize };
  CUDA_DRIVER_CHECK(UPLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, 0, args, 0));
  CUDACHECK(UPTKDeviceSynchronize());

  int h_warpSize;
  CUDA_DRIVER_CHECK(UPMemcpyDtoH(&h_warpSize, reinterpret_cast<UPTKdeviceptr>(d_warpSize), sizeof(int)));
  CUDA_DRIVER_CHECK(UPMemFree((UPTKdeviceptr)d_warpSize));
  CUDA_DRIVER_CHECK(UPModuleUnload(module));
  // Verifies warp size returned by the kernel via nvrtc and runtime to be same
  //REQUIRE(h_warpSize == props->warpSize);
  EXPECT_EQ(h_warpSize, props.warpSize);
}
