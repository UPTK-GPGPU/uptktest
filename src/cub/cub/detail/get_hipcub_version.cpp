// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "get_hipcub_version.hpp"

#define CUDA_CHECK(condition)                                                           \
    {                                                                                  \
        cudaError_t error = condition;                                                  \
        if(error != cudaSuccess)                                                        \
        {                                                                              \
            std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
            exit(error);                                                               \
        }                                                                              \
    }

__global__
void get_version_kernel(unsigned int * version)
{
    *version = HIPCUB_VERSION;
}

unsigned int get_hipcub_version_on_device()
{
    unsigned int version = 0;

    unsigned int * d_version;
    CUDA_CHECK(cudaMalloc(&d_version, sizeof(unsigned int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    get_version_kernel<<<
        dim3(1), dim3(1), 0, 0>>>(
        d_version
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(
            &version, d_version,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        )
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_version));

    return version;
}
