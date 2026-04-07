// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CUDA_CHECK(error)         \
  {                                  \
    if(error != cudaSuccess){         \
        std::cout << error << std::endl; \
        exit(error); \
    } \
  }

int main(int, char**)
{
    using T = unsigned int;

    // host input/output
    const size_t size = 1024 * 256;
    std::vector<T> input(size, 1);
    T output = 0;

    // device input/output
    T * d_input;
    T * d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(T)));
    CUDA_CHECK(
        cudaMemcpy(
            d_input, input.data(),
            input.size() * sizeof(T),
            cudaMemcpyHostToDevice
        )
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate expected results on host
    auto expected = std::accumulate(input.begin(), input.end(), 0U);

    // Temporary storage
    size_t temp_storage_size_bytes;
    void * d_temp_storage = nullptr;
    // Get size of d_temp_storage
    CUDA_CHECK(
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_size_bytes,
            d_input, d_output, input.size()
        )
    );

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_size_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run
    CUDA_CHECK(
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_size_bytes,
            d_input, d_output, input.size()
        )
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output to host
    CUDA_CHECK(
        cudaMemcpy(
            &output, d_output,
            sizeof(T),
            cudaMemcpyDeviceToHost
        )
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    if(output != expected)
    {
        std::cout
            << "Failure: output (" << output
            << ") != expected (" << expected << ")"
            << std::endl;
        return 1;
    }
    return 0;
}
