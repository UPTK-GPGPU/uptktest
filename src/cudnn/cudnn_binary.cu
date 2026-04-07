// ====------ cudnn-binary.cu ---------- *- CUDA -* ----===////
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
void cudnn_binary_test1()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i * 0.5f + 5.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }
    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));
    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    cudnnOpTensorDescriptor_t OpDesc;
    CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&OpDesc));
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;
    CUDNN_CHECK(cudnnOpTensor(
        handle,
        OpDesc,
        &alpha0,
        outTensor,
        out,
        &alpha1,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        12.5, 17.25, 22, 26.75, 31.5,
        36.25, 41, 45.75, 50.5, 55.25,
        60, 64.75, 69.5, 74.25, 79,
        83.75, 88.5, 93.25, 98, 102.75,
        107.5, 112.25, 117, 121.75, 126.5,

        131.25, 136, 140.75, 145.5, 150.25,
        155, 159.75, 164.5, 169.25, 174,
        178.75, 183.5, 188.25, 193, 197.75,
        202.5, 207.25, 212, 216.75, 221.5,
        226.25, 231, 235.75, 240.5, 245.25,

        250, 254.75, 259.5, 264.25, 269,
        273.75, 278.5, 283.25, 288, 292.75,
        297.5, 302.25, 307, 311.75, 316.5,
        321.25, 326, 330.75, 335.5, 340.25,
        345, 349.75, 354.5, 359.25, 364,

        368.75, 373.5, 378.25, 383, 387.75,
        392.5, 397.25, 402, 406.75, 411.5,
        416.25, 421, 425.75, 430.5, 435.25,
        440, 444.75, 449.5, 454.25, 459,
        463.75, 468.5, 473.25, 478, 482.75};
    check(expect, host_out, expect.size(), 1e-1, "cudnn_binary_test1");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(OpDesc));
    
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}

template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_binary_test2()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i * 0.5f + 5.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }
    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));
    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    cudnnOpTensorDescriptor_t OpDesc;
    CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&OpDesc));
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;
    CUDNN_CHECK(cudnnOpTensor(
        handle,
        OpDesc,
        &alpha0,
        outTensor,
        out,
        &alpha1,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        0, 22.625, 49, 79.125, 113,
        150.625, 192, 237.125, 286, 338.625,
        395, 455.125, 519, 586.625, 658,
        733.125, 812, 894.625, 981, 1071.12,
        1165, 1262.62, 1364, 1469.12, 1578,

        1690.62, 1807, 1927.12, 2051, 2178.62,
        2310, 2445.12, 2584, 2726.62, 2873,
        3023.12, 3177, 3334.62, 3496, 3661.12,
        3830, 4002.62, 4179, 4359.12, 4543,
        4730.62, 4922, 5117.12, 5316, 5518.62,

        5725, 5935.12, 6149, 6366.62, 6588,
        6813.12, 7042, 7274.62, 7511, 7751.12,
        7995, 8242.62, 8494, 8749.12, 9008,
        9270.62, 9537, 9807.12, 10081, 10358.6,
        10640, 10925.1, 11214, 11506.6, 11803,

        12103.1, 12407, 12714.6, 13026, 13341.1,
        13660, 13982.6, 14309, 14639.1, 14973,
        15310.6, 15652, 15997.1, 16346, 16698.6,
        17055, 17415.1, 17779, 18146.6, 18518,
        18893.1, 19272, 19654.6, 20041, 20431.1};
    check(expect, host_out, expect.size(), 1e-1, "cudnn_binary_test2");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(OpDesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}
template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_binary_test3()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i * 0.5f + 5.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }
    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));
    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    cudnnOpTensorDescriptor_t OpDesc;
    CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&OpDesc));
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_MIN, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;
    CUDNN_CHECK(cudnnOpTensor(
        handle,
        OpDesc,
        &alpha0,
        outTensor,
        out,
        &alpha1,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        0, 3.5, 7, 10.5, 14,
        17.5, 21, 24.5, 28, 31.5,
        35, 38.5, 42, 45.5, 49,
        52.5, 56, 59.5, 63, 66.5,
        70, 73.5, 77, 80.5, 84,

        87.5, 91, 94.5, 98, 101.5,
        105, 108.5, 112, 115.5, 119,
        122.5, 126, 129.5, 133, 136.5,
        140, 143.5, 147, 150.5, 154,
        157.5, 161, 164.5, 168, 171.5,

        175, 178.25, 181.5, 184.75, 188,
        191.25, 194.5, 197.75, 201, 204.25,
        207.5, 210.75, 214, 217.25, 220.5,
        223.75, 227, 230.25, 233.5, 236.75,
        240, 243.25, 246.5, 249.75, 253,

        256.25, 259.5, 262.75, 266, 269.25,
        272.5, 275.75, 279, 282.25, 285.5,
        288.75, 292, 295.25, 298.5, 301.75,
        305, 308.25, 311.5, 314.75, 318,
        321.25, 324.5, 327.75, 331, 334.25};
    check(expect, host_out, expect.size(), 1e-1, "cudnn_binary_test3");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(OpDesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}
template <cudnnDataType_t T, typename HT = typename dt_trait<T>::type>
void cudnn_binary_test4()
{
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t dataTensor, outTensor;
    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outTensor));

    int in = 2, ic = 2, ih = 5, iw = 5;
    int on = 2, oc = 2, oh = 5, ow = 5;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

    HT *data, *out;
    std::vector<HT> host_data(in * ic * ih * iw, 1.0f);
    std::vector<HT> host_out(on * oc * oh * ow, 0.0f);

    for (int i = 0; i < in * ic * ih * iw; i++)
    {
        host_data[i] = i * 0.5f + 5.f;
    }
    for (int i = 0; i < on * oc * oh * ow; i++)
    {
        host_out[i] = i;
    }
    CUDA_CHECK(cudaMalloc(&data, sizeof(HT) * in * ic * ih * iw));
    CUDA_CHECK(cudaMalloc(&out, sizeof(HT) * on * oc * oh * ow));
    CUDA_CHECK(cudaMemcpy(data, host_data.data(), sizeof(HT) * in * ic * ih * iw, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out, host_out.data(), sizeof(HT) * on * oc * oh * ow, cudaMemcpyHostToDevice));

    cudnnOpTensorDescriptor_t OpDesc;
    CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&OpDesc));
    CUDNN_CHECK(cudnnSetOpTensorDescriptor(OpDesc, CUDNN_OP_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

    float alpha0 = 1.5f, alpha1 = 2.5f, beta = 2.f;
    CUDNN_CHECK(cudnnOpTensor(
        handle,
        OpDesc,
        &alpha0,
        outTensor,
        out,
        &alpha1,
        dataTensor,
        data,
        &beta,
        outTensor,
        out));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out.data(), out, sizeof(HT) * on * oc * oh * ow, cudaMemcpyDeviceToHost));
    std::vector<float> expect = {
        12.5, 15.75, 19, 22.25, 25.5,
        28.75, 32, 35.25, 38.5, 41.75,
        45, 48.25, 51.5, 54.75, 58,
        61.25, 64.5, 67.75, 71, 74.25,
        77.5, 80.75, 84, 87.25, 90.5,

        93.75, 97, 100.25, 103.5, 106.75,
        110, 113.25, 116.5, 119.75, 123,
        126.25, 129.5, 132.75, 136, 139.25,
        142.5, 145.75, 149, 152.25, 155.5,
        158.75, 162, 165.25, 168.5, 171.75,

        175, 178.5, 182, 185.5, 189,
        192.5, 196, 199.5, 203, 206.5,
        210, 213.5, 217, 220.5, 224,
        227.5, 231, 234.5, 238, 241.5,
        245, 248.5, 252, 255.5, 259,

        262.5, 266, 269.5, 273, 276.5,
        280, 283.5, 287, 290.5, 294,
        297.5, 301, 304.5, 308, 311.5,
        315, 318.5, 322, 325.5, 329,
        332.5, 336, 339.5, 343, 346.5};
    check(expect, host_out, expect.size(), 1e-1, "cudnn_binary_test4");

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outTensor));
    CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(OpDesc));

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(out));
}

TEST(cudnn, cudnn_binary)
{
    cudnn_binary_test1<CUDNN_DATA_FLOAT>();
    cudnn_binary_test2<CUDNN_DATA_FLOAT>();
    cudnn_binary_test3<CUDNN_DATA_FLOAT>();
    cudnn_binary_test4<CUDNN_DATA_FLOAT>();
}
