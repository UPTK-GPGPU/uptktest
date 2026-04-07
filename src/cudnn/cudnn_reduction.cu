// ====------ CUDNN_CHECK(cudnn-reduction.cu ---------- *- CUDA -* ----===////
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
void cudnn_reduction_test1()
{

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 2, oh = 6, ow = 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i - 25.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    float alpha = 2.5f, beta = 1.5f;

    cudnnReduceTensorDescriptor_t reducedesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reducedesc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));
    size_t ws_size;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size));
    void *ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));

    CUDNN_CHECK(cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {-337.5, -246, -154.5, -63, 28.5, 120,
                                 211.5, 303, 394.5, 486, 577.5, 669,
                                 760.5, 852, 943.5, 1035, 1126.5, 1218,
                                 1309.5, 1401, 1492.5, 1584, 1675.5, 1767};
    check(expect, host_out, expect.size(), 1e-3, "cudnn_reduction_test1");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reducedesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(ws));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_reduction_test2()
{

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i - 25.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    float alpha = 2.5f, beta = 1.5f;

    cudnnReduceTensorDescriptor_t reducedesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reducedesc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));
    size_t ws_size;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size));
    void *ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));

    CUDNN_CHECK(cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        -135, 46.5, 228, 409.5, 591, 772.5,
        2034, 2215.5, 2397, 2578.5, 2760, 2941.5};
    check(expect, host_out, expect.size(), 1e-3, "cudnn_reduction_test2");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reducedesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(ws));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_reduction_test3()
{

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i - 25.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    float alpha = 2.5f, beta = 1.5f;

    cudnnReduceTensorDescriptor_t reducedesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reducedesc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_NORM1,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));
    size_t ws_size;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size));
    void *ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));

    CUDNN_CHECK(cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        540, 541.5, 543, 544.5, 596, 772.5,
        2034, 2215.5, 2397, 2578.5, 2760, 2941.5};
    check(expect, host_out, expect.size(), 1e-3, "cudnn_reduction_test3");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reducedesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(ws));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_reduction_test4()
{

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 6, iw = 6;
    int on = 2, oc = 1, oh = 6, ow = 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, T, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, T, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * iw * ih, 0);
    std::vector<HT> host_out(on * oc * ow * oh, 0);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i - 25.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));

    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    float alpha = 2.5f, beta = 1.5f;

    cudnnReduceTensorDescriptor_t reducedesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reducedesc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reducedesc,
        CUDNN_REDUCE_TENSOR_NORM2,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));
    size_t ws_size;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        handle,
        reducedesc,
        dataTensor,
        outTensor,
        &ws_size));
    void *ws;
    CUDA_CHECK(cudaMalloc(&ws, ws_size));

    CUDNN_CHECK(cudnnReduceTensor(
        handle,
        reducedesc,
        0,
        0,
        ws,
        ws_size,
        &alpha,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        161.361, 158.623, 172.521, 199.916,
        236.299, 278.217, 614.176, 666.005,
        718.072, 770.327, 822.736, 875.271};
    check(expect, host_out, expect.size(), 1e-3, "cudnn_reduction_test4");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reducedesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(ws));
}

TEST(cudnn, cudnn_reduction)
{
    cudnn_reduction_test1<CUDNN_DATA_FLOAT>();
    cudnn_reduction_test2<CUDNN_DATA_FLOAT>();
    cudnn_reduction_test3<CUDNN_DATA_FLOAT>();
    cudnn_reduction_test4<CUDNN_DATA_FLOAT>();
}
