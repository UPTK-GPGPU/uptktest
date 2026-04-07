// ====------ cudnn-fill.cu ---------- *- CUDA -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include "cudnn_utils.h"

template <cudnnDataType_t T>
struct dt_trait
{
    typedef void type;
};
template <>
struct dt_trait<CUDNN_DATA_FLOAT>
{
    typedef float type;
};
template <>
struct dt_trait<CUDNN_DATA_INT32>
{
    typedef int type;
};
template <>
struct dt_trait<CUDNN_DATA_HALF>
{
    typedef float type;
};

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_fill_test()
{

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;
    HT *data;
    HT value = 1.5;
    std::vector<HT> host_data(ele_num, 0);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));

    CUDNN_CHECK(cudnnSetTensor(handle, dataTensor, data, &value));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_data.data(), data, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));
    float precision = 1e-3;
    for (int i = 0; i < ele_num; i++)
    {
        if (std::abs(host_data[i] - value) > precision)
        {
            EXPECT_NEAR(host_data[i], value, precision);
            break;
        }
    }
    
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDA_CHECK(cudaFree(data));

}

TEST(cudnn, cudnn_fill)
{
    cudnn_fill_test<CUDNN_DATA_FLOAT>();
}