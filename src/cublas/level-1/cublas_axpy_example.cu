/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <UPTK_blas.h>
#include <UPTK_runtime.h>

#include "../utils/cublas_utils.h"
#include <gtest/gtest.h>   


using data_type = double;

TEST(cublas_level_1,cublas_axpy_example) {
    UPTKblasHandle_t cublasH = NULL;
    UPTKStream_t stream = NULL;

    /*
     *   A = | 1.0 2.0 3.0 4.0 |
     *   B = | 5.0 6.0 7.0 8.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    const data_type alpha = 2.1;
    const int incx = 1;
    const int incy = 1;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    // printf("A\n");
    // check_vector(A.size(), A.data());
    // printf("=====\n");

    // printf("B\n");
    // check_vector(B.size(), B.data());
    // printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(UPTKblasCreate(&cublasH));

    CUDA_CHECK(UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking));
    CUBLAS_CHECK(UPTKblasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(UPTKMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(UPTKMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK(UPTKMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), UPTKMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(UPTKMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), UPTKMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(UPTKblasDaxpy(cublasH, A.size(), &alpha, d_A, incx, d_B, incy));

    /* step 4: copy data to host */
    CUDA_CHECK(UPTKMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), UPTKMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(UPTKStreamSynchronize(stream));

    /*
     *   B = | 7.10 10.20 13.30 16.40 |
     */
    double eps = 1.0E-2;
    const std::vector<data_type> expect = {7.10 ,10.20 ,13.30 ,16.40};
    //printf("B\n");
    check_vector(B.size(), B.data(), expect.data(), eps);
    //printf("=====\n");

    /* free resources */
    CUDA_CHECK(UPTKFree(d_A));
    CUDA_CHECK(UPTKFree(d_B));

    CUBLAS_CHECK(UPTKblasDestroy(cublasH));

    CUDA_CHECK(UPTKStreamDestroy(stream));

    CUDA_CHECK(UPTKDeviceReset());

    //return EXIT_SUCCESS;
}
