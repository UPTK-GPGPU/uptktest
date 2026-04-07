// ====------ cudnn-softmax.cu ---------- *- CUDA -* ----===////
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
void cudnn_softmax_test1()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));

    HT *data, *out;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);

    for (int i = 0; i < ele_num; i++)
    {
        host_data[i] = 10 * i;
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&out, ele_num * sizeof(HT)));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));

    float alpha = 2.f, beta = 1.5f;
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, dataTensor, data, &beta, outTensor, out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));

    std::vector<float> expect = {
        0, 1.5, 3, 4.5, 6,
        7.5, 9, 10.5, 12, 13.5,
        15, 16.5, 18, 19.5, 21,
        22.5, 24, 25.5, 27, 28.5,
        30, 31.5, 33, 34.5, 36,
        39.5, 41, 42.5, 44, 45.5,
        47, 48.5, 50, 51.5, 53,
        54.5, 56, 57.5, 59, 60.5,
        62, 63.5, 65, 66.5, 68,
        69.5, 71, 72.5, 74, 75.5};
    check(expect, host_out, expect.size(), 1e-3, "cudnn_softmax_test1");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_softmax_test2()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffdataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffoutTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num);
    for (int i = 0; i < ele_num; i++)
    {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&out, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffdata, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffout, ele_num * sizeof(HT)));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffout, host_diffout.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));

    float alpha = 1.5f, beta = 0.f;
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, dataTensor, data, &beta, outTensor, out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));
    alpha = 2.f, beta = 0.f;
    CUDNN_CHECK(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outTensor, out, diffoutTensor, diffout, &beta, diffdataTensor, diffdata));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));

    std::vector<float> expect = {
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -0.113787, -0.113787, -0.113787, -0.113787, -0.113787,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621,
        -1.38621, -1.38621, -1.38621, -1.38621, -1.38621};
    check(expect, host_diffdata, expect.size(), 1e-3, "cudnn_softmax_test2");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffdataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffoutTensor));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(diffdata));
    CUDA_CHECK(cudaFree(diffout));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_softmax_test3()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor, diffdataTensor, diffoutTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffdataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&diffoutTensor));
    int n = 1, c = 2, h = 5, w = 5;
    int ele_num = n * c * h * w;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffdataTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(diffoutTensor, CUDNN_TENSOR_NCHW, T, n, c, h, w));
    HT *data, *out, *diffdata, *diffout;
    std::vector<HT> host_data(ele_num);
    std::vector<HT> host_out(ele_num);
    std::vector<HT> host_diffdata(ele_num);
    std::vector<HT> host_diffout(ele_num);
    for (int i = 0; i < ele_num; i++)
    {
        host_data[i] = i * 0.1f;
        host_out[i] = i;
        host_diffdata[i] = i;
        host_diffout[i] = 1.f;
    }

    CUDA_CHECK(cudaMalloc(&data, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&out, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffdata, ele_num * sizeof(HT)));
    CUDA_CHECK(cudaMalloc(&diffout, ele_num * sizeof(HT)));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffdata, host_diffdata.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(diffout, host_diffout.data(), ele_num * sizeof(HT), cudaMemcpyHostToDevice));

    float alpha = 1.5f, beta = 0.f;
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, dataTensor, data, &beta, outTensor, out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));
    alpha = 2.f, beta = 1.5f;
    CUDNN_CHECK(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, outTensor, out, diffoutTensor, diffout, &beta, diffdataTensor, diffdata));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_diffdata.data(), diffdata, ele_num * sizeof(HT), cudaMemcpyDeviceToHost));

    std::vector<float> expect = {
        -0.113787, 1.38621, 2.88621, 4.38621, 5.88621,
        7.38621, 8.88621, 10.3862, 11.8862, 13.3862,
        14.8862, 16.3862, 17.8862, 19.3862, 20.8862,
        22.3862, 23.8862, 25.3862, 26.8862, 28.3862,
        29.8862, 31.3862, 32.8862, 34.3862, 35.8862,
        36.1138, 37.6138, 39.1138, 40.6138, 42.1138,
        43.6138, 45.1138, 46.6138, 48.1138, 49.6138,
        51.1138, 52.6138, 54.1138, 55.6138, 57.1138,
        58.6138, 60.1138, 61.6138, 63.1138, 64.6138,
        66.1138, 67.6138, 69.1138, 70.6138, 72.1138};
    check(expect, host_diffdata, expect.size(), 1e-3, "cudnn_softmax_test3");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffdataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(diffoutTensor));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(diffdata));
    CUDA_CHECK(cudaFree(diffout));
}

TEST(cudnn, cudnn_softmax)
{
    cudnn_softmax_test1<CUDNN_DATA_FLOAT>();
    cudnn_softmax_test2<CUDNN_DATA_FLOAT>();
    cudnn_softmax_test3<CUDNN_DATA_FLOAT>();
}