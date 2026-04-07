// ====------ CUDNN_CHECK(cudnn-memory.cu ---------- *- CUDA -* ----===////
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
void check(std::vector<T> &expect, std::vector<T> &actual, int num, const char *testName)
{
  for (int i = 0; i < num; i++)
  {
    if (expect[i] != actual[i])
    {
      EXPECT_EQ(expect[i], actual[i]);
      printf("test name = %s, index = %d\n", testName, i);
      break;
    }
  }
}

void cudnn_memory_test()
{

  cudnnHandle_t handle;
  cudnnTensorDescriptor_t dataTensor;

  CUDNN_CHECK(cudnnCreate(&handle));

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dataTensor));

  int on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride;
  size_t size;
  cudnnDataType_t odt;
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(dataTensor, CUDNN_DATA_FLOAT, 1, 2, 5, 5, 50, 25, 5, 1));
  CUDNN_CHECK(cudnnGetTensor4dDescriptor(dataTensor, &odt, &on, &oc, &oh, &ow, &on_stride, &oc_stride, &oh_stride, &ow_stride));
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(dataTensor, &size));
  std::vector<int> expect3 = {1, 2, 5, 5, 50, 25, 5, 1, 200};
  std::vector<int> actual3 = {on, oc, oh, ow, on_stride, oc_stride, oh_stride, ow_stride, (int)size};
  check<int>(expect3, actual3, expect3.size(), "cudnn_memory_test1");

  int dims[4] = {1, 4, 5, 5};
  int odims[4] = {0, 0, 0, 0};
  int strides[4] = {100, 25, 5, 1};
  int ostrides[4] = {0, 0, 0, 0};
  int ndims = 4, r_ndims = 4, ondims = 0;

  // Test 4
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(dataTensor, CUDNN_DATA_FLOAT, ndims, dims, strides));
  CUDNN_CHECK(cudnnGetTensorNdDescriptor(dataTensor, r_ndims, &odt, &ondims, odims, ostrides));
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(dataTensor, &size));
  std::vector<int> expect4 = {4, 1, 4, 5, 5, 100, 25, 5, 1, 400};
  std::vector<int> actual4 = {ondims, odims[0], odims[1], odims[2], odims[3], ostrides[0],
                              ostrides[1], ostrides[2], ostrides[3], (int)size};
  check<int>(expect4, actual4, expect4.size(), "cudnn_memory_test2");

  CUDNN_CHECK(cudnnDestroy(handle));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(dataTensor));
}

TEST(cudnn, cudnn_memory)
{
  cudnn_memory_test();
}