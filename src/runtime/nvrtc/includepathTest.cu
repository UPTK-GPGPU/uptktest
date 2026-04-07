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
#include <fstream>

static constexpr auto NUM_THREADS{128};
static constexpr auto NUM_BLOCKS{32};

// This test verifies hiprtc compilation by passing include path option using -I with spaces
// before the path. eg: -I   ../ or -I /path/to/headers etc.

TEST(cudanvrtc, includepathTest) {
  using namespace std;

  string saxpy = "";
  {
    fstream f("saxpy.h");
    if (f.is_open()) {
      size_t sizeFile;
      f.seekg(0, fstream::end);
      size_t size = sizeFile = (size_t)f.tellg();
      f.seekg(0, fstream::beg);
      saxpy.resize(size, ' ');
      f.read(&saxpy[0], size);
      f.close();
   }
 }

  nvrtcProgram prog;
  nvrtcCreateProgram(&prog,         // prog
                      saxpy.c_str(), // buffer
                      "saxpy.cu",    // name
                      0, nullptr, nullptr);

  UPTKDeviceProp props;
  int device = 0;
  CUDACHECK(UPTKGetDeviceProperties(&props, device));
  string sarg = string("--gpu-architecture=compute_")
    + to_string(props.major) + to_string(props.minor);
  // Need to find the header files from the include path
  // It is set to headers in the current directory here
  const char* options[] = {
      sarg.c_str(), "-I./headers"
  };

  nvrtcResult compileResult{nvrtcCompileProgram(prog, 2, options)};
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

  vector<char> code(codeSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, code.data()));

  NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  // Do hip malloc first so that we donot need to do a UPInit manually before calling hipModule APIs
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);

  float *dX, *dY, *dOut;
  CUDA_DRIVER_CHECK(UPMemAlloc((UPTKdeviceptr*)&dX, bufferSize));
  CUDA_DRIVER_CHECK(UPMemAlloc((UPTKdeviceptr*)&dY, bufferSize));
  CUDA_DRIVER_CHECK(UPMemAlloc((UPTKdeviceptr*)&dOut, bufferSize));

  UPTKmodule module;
  UPTKfunction kernel;
  CUDA_DRIVER_CHECK(UPModuleLoadData(&module, code.data()));
  CUDA_DRIVER_CHECK(UPModuleGetFunction(&kernel, module, "saxpy"));

  float a = 5.1f;
  unique_ptr<float[]> hX{new float[n]};
  unique_ptr<float[]> hY{new float[n]};
  unique_ptr<float[]> hOut{new float[n]};
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  CUDA_DRIVER_CHECK(UPMemcpyHtoD((UPTKdeviceptr)dX, hX.get(), bufferSize));
  CUDA_DRIVER_CHECK(UPMemcpyHtoD((UPTKdeviceptr)dY, hY.get(), bufferSize));

  struct {
    float a_;
    float* b_;
    float* c_;
    float* d_;
    size_t e_;
  } args{a, dX, dY, dOut, n};

  auto size = sizeof(args);
  void* config[] = {UPTK_LAUNCH_PARAM_BUFFER_POINTER, &args, UPTK_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    UPTK_LAUNCH_PARAM_END};
  CUDA_DRIVER_CHECK(UPLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1, 0, nullptr, nullptr, config));

  CUDA_DRIVER_CHECK(UPMemcpyDtoH(hOut.get(), (UPTKdeviceptr)dOut, bufferSize));

  CUDA_DRIVER_CHECK(UPMemFree((UPTKdeviceptr)dX));
  CUDA_DRIVER_CHECK(UPMemFree((UPTKdeviceptr)dY));
  CUDA_DRIVER_CHECK(UPMemFree((UPTKdeviceptr)dOut));

  CUDA_DRIVER_CHECK(UPModuleUnload(module));

  for (size_t i = 0; i < n; ++i) {
    EXPECT_LE(fabs(a * hX[i] + hY[i] - hOut[i]), fabs(hOut[i]) * 1e-6);
  }
}
