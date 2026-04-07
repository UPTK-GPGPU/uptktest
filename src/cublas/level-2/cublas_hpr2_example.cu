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

using data_type = cuDoubleComplex;

TEST(cublas_level_2,cublas_hpr2_example) {
    UPTKblasHandle_t cublasH = NULL;
    UPTKStream_t stream = NULL;

    const int m = 2;
    const int n = 2;
    const int lda = m;

    /*
     *   AP = | 1.1 + 1.2j | 2.3 + 2.4j |
     *        | 3.5 + 3.6j | 4.7 + 4.8j |
     *   x  = | 5.1 + 6.2j | 7.3 + 8.4j |
     *   y  = | 1.1 + 2.2j | 3.3 + 4.4j |
     */

    std::vector<data_type> AP = {{1.1, 1.2}, {3.5, 3.6}, {2.3, 2.4}, {4.7, 4.8}};
    const std::vector<data_type> x = {{5.1, 6.2}, {7.3, 8.4}};
    const std::vector<data_type> y = {{1.1, 2.2}, {3.3, 4.4}};
    const data_type alpha = {1.0, 1.0};
    const int incx = 1;
    const int incy = 1;

    data_type *d_AP = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    UPTKblasFillMode_t uplo = UPTKBLAS_FILL_MODE_UPPER;

    // printf("AP\n");
    // check_matrix(m, n, AP.data(), lda);
    // printf("=====\n");

    // printf("x\n");
    // check_vector(x.size(), x.data());
    // printf("=====\n");

    // printf("y\n");
    // check_vector(y.size(), y.data());
    // printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(UPTKblasCreate(&cublasH));

    CUDA_CHECK(UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking));
    CUBLAS_CHECK(UPTKblasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(UPTKMalloc(reinterpret_cast<void **>(&d_AP), sizeof(data_type) * AP.size()));
    CUDA_CHECK(UPTKMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(UPTKMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    CUDA_CHECK(UPTKMemcpyAsync(d_AP, AP.data(), sizeof(data_type) * AP.size(),
                               UPTKMemcpyHostToDevice, stream));
    CUDA_CHECK(UPTKMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), UPTKMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(UPTKMemcpyAsync(d_y, y.data(), sizeof(data_type) * y.size(), UPTKMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(UPTKblasZhpr2(cublasH, uplo, n, &alpha, d_x, incx, d_y, incy, d_AP));

    /* step 4: copy data to host */
    CUDA_CHECK(UPTKMemcpyAsync(AP.data(), d_AP, sizeof(data_type) * AP.size(),
                               UPTKMemcpyDeviceToHost, stream));

    CUDA_CHECK(UPTKStreamSynchronize(stream));

    /*
     *   AP = | 48.40 +  0.00j | 133.20 + 0.00j |
     *        | 82.92 + 26.04j |   4.70 + 4.80j |
     */
    double eps = 1.0E-2;
    const std::vector<data_type> expect = {{48.40, 0.00}, {82.92, 26.04},{133.20, 0.00}, {4.70, 4.80}};
    //printf("AP\n");
    check_matrix(m, n, AP.data(), lda, expect.data(), eps);
    //printf("=====\n");

    /* free resources */
    CUDA_CHECK(UPTKFree(d_AP));
    CUDA_CHECK(UPTKFree(d_x));
    CUDA_CHECK(UPTKFree(d_y));

    CUBLAS_CHECK(UPTKblasDestroy(cublasH));

    CUDA_CHECK(UPTKStreamDestroy(stream));

    CUDA_CHECK(UPTKDeviceReset());

    //return EXIT_SUCCESS;
}
