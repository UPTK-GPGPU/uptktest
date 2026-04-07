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

#include "cufft_utils.h"

TEST(cufft, 1d_c2c_example)
{
    UPTKfftHandle plan;
    UPTKStream_t stream = NULL;

    int n = 8;
    int batch_size = 2;
    int fft_size = batch_size * n;

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(fft_size);

    for (int i = 0; i < fft_size; i++)
    {
        data[i] = data_type(i, -i);
    }

    UPTKfftComplex *d_data = nullptr;

    CUFFT_CALL(UPTKfftCreate(&plan));
    CUFFT_CALL(UPTKfftPlan1d(&plan, data.size(), UPTKFFT_C2C, batch_size));

    CUDA_RT_CALL(UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking));
    CUFFT_CALL(UPTKfftSetStream(plan, stream));

    // Create device data arrays
    CUDA_RT_CALL(UPTKMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * data.size()));
    CUDA_RT_CALL(UPTKMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
                                 UPTKMemcpyHostToDevice, stream));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    CUFFT_CALL(UPTKfftExecC2C(plan, d_data, d_data, UPTKFFT_FORWARD));
    CUFFT_CALL(UPTKfftExecC2C(plan, d_data, d_data, UPTKFFT_INVERSE));

    CUDA_RT_CALL(UPTKMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
                                 UPTKMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(UPTKStreamSynchronize(stream));

    std::vector<data_type> expect_data(fft_size);
    expect_data[0] = data_type(0.f, 0.f);
    expect_data[1] = data_type(16.f, -16.f);
    expect_data[2] = data_type(32.f, -32.f);
    expect_data[3] = data_type(48.f, -48.f);
    expect_data[4] = data_type(64.f, -64.f);
    expect_data[5] = data_type(80.f, -80.f);
    expect_data[6] = data_type(96.f, -96.f);
    expect_data[7] = data_type(112.f, -112.f);
    expect_data[8] = data_type(128.f, -128.f);
    expect_data[9] = data_type(144.f, -144.f);
    expect_data[10] = data_type(160.f, -160.f);
    expect_data[11] = data_type(176.f, -176.f);
    expect_data[12] = data_type(192.f, -192.f);
    expect_data[13] = data_type(208.f, -208.f);
    expect_data[14] = data_type(224.f, -224.f);
    expect_data[15] = data_type(240.f, -240.f);

    for (int i = 0; i < fft_size; i++)
    {
        EXPECT_NEAR(data[i].real(), expect_data[i].real(), 0.01);
        EXPECT_NEAR(data[i].imag(), expect_data[i].imag(), 0.01);
    }

    /* free resources */
    CUDA_RT_CALL(UPTKFree(d_data))

    CUFFT_CALL(UPTKfftDestroy(plan));

    CUDA_RT_CALL(UPTKStreamDestroy(stream));

    CUDA_RT_CALL(UPTKDeviceReset());
}