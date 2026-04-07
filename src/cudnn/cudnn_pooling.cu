// ====------ cudnn-pooling.cu ---------- *- CUDA -* ----===////
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
#include <stdio.h>
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

template <typename T>
void check(std::vector<T> &expect, std::vector<T> &actual, int num, float precision, const char *testName)
{
    for (int i = 0; i < num; i++)
    {
        if (std::abs(expect[i] - actual[i]) > precision)
        {
            EXPECT_NEAR(expect[i], actual[i], precision);
            printf("test name = %s, index = %d\n", testName, i);
            break;
        }
    }
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_pooling_test1()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnPoolingDescriptor_t desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    int ele_num2 = on * oc * oh * ow;
    std::vector<HT> host_out(ele_num2);

    for (int i = 0; i < ele_num; i++)
    {
        host_data[i] = i;
    }

    for (int i = 0; i < ele_num2; i++)
    {
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&out, ele_num2 * sizeof(HT)));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice));

    float alpha = 1.f, beta = 0.f;
    CUDNN_CHECK(cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(HT), cudaMemcpyDeviceToHost));

    std::vector<float> expect = {
        0, 2, 4, 4,
        10, 12, 14, 14,
        20, 22, 24, 24,
        20, 22, 24, 24,
        25, 27, 29, 29,
        35, 37, 39, 39,
        45, 47, 49, 49,
        45, 47, 49, 49};

    check(expect, host_out, expect.size(), 1e-3, "cudnn_pooling_test1");
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_pooling_test2()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnPoolingDescriptor_t desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 4, 4, 3, 3, 2, 2));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffdataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffoutTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));

    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(desc, dataTensor, &on, &oc, &oh, &ow));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));
    int ele_num2 = on * oc * oh * ow;

    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num2);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num2);
    for (int i = 0; i < ele_num; i++)
    {
        host_data[i] = i * 0.1f;
        host_diffdata[i] = i;
    }
    for (int i = 0; i < ele_num2; i++)
    {
        host_out[i] = i;
        host_diffout[i] = 1.f;
    }

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&out, ele_num2 * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffdata, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffout, ele_num2 * sizeof(HT)));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffout, host_diffout.data(), ele_num2 * sizeof(HT), cudaMemcpyHostToDevice));

    float alpha = 1.5f, beta = 1.f;
    CUDNN_CHECK(cudnnPoolingForward(handle, desc, &alpha, dataTensor, data, &beta, outTensor, out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, ele_num2 * sizeof(HT), cudaMemcpyDeviceToHost));
    alpha = 1.5f, beta = 1.f;
    CUDNN_CHECK(cudnnPoolingBackward(handle, desc, &alpha, outTensor, out, diffoutTensor, diffout, dataTensor, data, &beta, diffdataTensor, diffdata));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));

    std::vector<float> expect = {
        1.5, 1, 3.5, 3, 7,
        5, 6, 7, 8, 9,
        11.5, 11, 13.5, 13, 17,
        15, 16, 17, 18, 19,
        23, 21, 25, 23, 30,
        26.5, 26, 28.5, 28, 32,
        30, 31, 32, 33, 34,
        36.5, 36, 38.5, 38, 42,
        40, 41, 42, 43, 44,
        48, 46, 50, 48, 55};

    check(expect, host_diffdata, expect.size(), 1e-3, "cudnn_pooling_test2");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffdataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffoutTensor));
    
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(diffdata));
    CUDA_CHECK(cudaFree(diffout));
}

TEST(cudnn, cudnn_pooling)
{
    cudnn_pooling_test1<CUDNN_DATA_FLOAT>();
    cudnn_pooling_test2<CUDNN_DATA_FLOAT>();
}