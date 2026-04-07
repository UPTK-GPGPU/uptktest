/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _CUBLAS_UTILS_H
#define _CUBLAS_UTILS_H

#include <gtest/gtest.h>   
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <UPTK_blas.h>
#include <UPTK_runtime_api.h>
#include <library_types.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        UPTKError_t err_ = (err);                                                                  \
        if (err_ != UPTKSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        UPTKblasStatus_t err_ = (err);                                                               \
        if (err_ != UPTKBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

// type traits
template <typename T>  struct traits;

template <>  struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr UPTKDataType cuda_data_type = UPTK_R_32F;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <>  struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr UPTKDataType cuda_data_type = UPTK_R_64F;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <>  struct traits<cuFloatComplex> {
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr UPTKDataType cuda_data_type = UPTK_C_32F;

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, double f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <>  struct traits<cuDoubleComplex> {
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr UPTKDataType cuda_data_type = UPTK_C_64F;

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, double f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};

template <typename T> static void check_matrix(const int &m, const int &n, const T *actual, const int &lda, const T *expect, const float eps );

template <>  void check_matrix(const int &m, const int &n, const float *actual, const int &lda, const float *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //std::printf("%0.2f ", actual[j * lda + i]);
            if (std::abs(actual[i] - expect[i]) > eps)
            {
                EXPECT_EQ(actual[j * lda + i], expect[j * lda + i]);
                break;
            }
        }
        //std::printf("\n");
    }
}

template <>  void check_matrix(const int &m, const int &n, const double *actual, const int &lda, const double *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //std::printf("%0.2f ", actual[j * lda + i]);
            if (std::abs(actual[j * lda + i] - expect[j * lda + i]) > eps)
            {
                EXPECT_EQ(actual[j * lda + i], expect[j * lda + i]);
                break;
            }
        }
        //std::printf("\n");
    }
}

template <>  void check_matrix(const int &m, const int &n, const cuComplex *actual, const int &lda, const cuComplex *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //std::printf("%0.2f + %0.2fj ", actual[j * lda + i].x, actual[j * lda + i].y);
            if (std::abs(actual[j * lda + i].x - expect[j * lda + i].x) > eps 
               || std::abs(actual[j * lda + i].y - expect[j * lda + i].y) > eps)
            {
                EXPECT_EQ(actual[j * lda + i].x, expect[j * lda + i].x);
                EXPECT_EQ(actual[j * lda + i].y, expect[j * lda + i].y);
                break;
            }
        }
        //std::printf("\n");
    }
}

template <>
 void check_matrix(const int &m, const int &n, const cuDoubleComplex *actual, const int &lda, const cuDoubleComplex *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            //std::printf("%0.2f + %0.2fj ", actual[j * lda + i].x, actual[j * lda + i].y);
            if (std::abs(actual[j * lda + i].x - expect[j * lda + i].x) > eps 
               || std::abs(actual[j * lda + i].y - expect[j * lda + i].y) > eps)
            {
                EXPECT_EQ(actual[j * lda + i].x, expect[j * lda + i].x);
                EXPECT_EQ(actual[j * lda + i].y, expect[j * lda + i].y);
                break;
            }
        }
        //std::printf("\n");
    }
}

template <typename T> static void check_vector(const int &m, const T *actual, const T *expect, const float eps );

template <>  void check_vector(const int &m, const float *actual, const float *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        //std::printf("%0.2f ", actual[i]);
        if (std::abs(actual[i] - expect[i]) > eps)
        {
            EXPECT_EQ(actual[i], expect[i]);
            break;
        }
    }
    //std::printf("\n");
}

template <>  void check_vector(const int &m, const double *actual, const double *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        //std::printf("%0.2f ", actual[i]);
        if (std::abs(actual[i] - expect[i]) > eps)
        {
            EXPECT_EQ(actual[i], expect[i]);
            break;
        }
    }
    //std::printf("\n");
}

template <>  void check_vector(const int &m, const cuComplex *actual, const cuComplex *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        //std::printf("%0.2f + %0.2fj ", actual[i].x, actual[i].y);
        if (std::abs(actual[i].x - expect[i].x) > eps 
            || std::abs(actual[i].y - expect[i].y) > eps)
        {
            EXPECT_EQ(actual[i].x, expect[i].x);
            EXPECT_EQ(actual[i].y, expect[i].y);
            break;
        }
    }
    //std::printf("\n");
}

template <>  void check_vector(const int &m, const cuDoubleComplex *actual, const cuDoubleComplex *expect, const float eps ) {
    for (int i = 0; i < m; i++) {
        //std::printf("%0.2f + %0.2fj ", actual[i].x, actual[i].y);
        if (std::abs(actual[i].x - expect[i].x) > eps 
            || std::abs(actual[i].y - expect[i].y) > eps)
        {
            EXPECT_EQ(actual[i].x, expect[i].x);
            EXPECT_EQ(actual[i].y, expect[i].y);
            break;
        }
    }
    //std::printf("\n");
}

template <typename T> static void generate_random_matrix(int m, int n, T **A, int *lda) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename traits<T>::S> dis(-1.0, 1.0);
    auto rand_gen = std::bind(dis, gen);

    *lda = n;

    size_t matrix_mem_size = static_cast<size_t>(*lda * m * sizeof(T));
    // suppress gcc 7 size warning
    if (matrix_mem_size <= PTRDIFF_MAX)
        *A = (T *)malloc(matrix_mem_size);
    else
        throw std::runtime_error("Memory allocation size is too large");

    if (*A == NULL)
        throw std::runtime_error("Unable to allocate host matrix");

    // random matrix and accumulate row sums
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T *A_row = (*A) + *lda * i;
            A_row[j] = traits<T>::rand(rand_gen);
        }
    }
}

// Makes matrix A of size mxn and leading dimension lda diagonal dominant
template <typename T> static void make_diag_dominant_matrix(int m, int n, T *A, int lda) {
    for (int i = 0; i < std::min(m, n); ++i) {
        T *A_row = A + lda * i;
        auto row_sum = traits<typename traits<T>::S>::zero;
        for (int j = 0; j < n; ++j) {
            row_sum += traits<T>::abs(A_row[j]);
        }
        A_row[i] = traits<T>::add(A_row[i], row_sum);
    }
}

// Returns UPTKDataType value as defined in library_types.h for the string
// containing type name
static UPTKDataType get_cuda_library_type(std::string type_string) {
    if (type_string.compare("UPTK_R_16F") == 0)
        return UPTK_R_16F;
    else if (type_string.compare("UPTK_C_16F") == 0)
        return UPTK_C_16F;
    else if (type_string.compare("UPTK_R_32F") == 0)
        return UPTK_R_32F;
    else if (type_string.compare("UPTK_C_32F") == 0)
        return UPTK_C_32F;
    else if (type_string.compare("UPTK_R_64F") == 0)
        return UPTK_R_64F;
    else if (type_string.compare("UPTK_C_64F") == 0)
        return UPTK_C_64F;
    else if (type_string.compare("UPTK_R_8I") == 0)
        return UPTK_R_8I;
    else if (type_string.compare("UPTK_C_8I") == 0)
        return UPTK_C_8I;
    else if (type_string.compare("UPTK_R_8U") == 0)
        return UPTK_R_8U;
    else if (type_string.compare("UPTK_C_8U") == 0)
        return UPTK_C_8U;
    else if (type_string.compare("UPTK_R_32I") == 0)
        return UPTK_R_32I;
    else if (type_string.compare("UPTK_C_32I") == 0)
        return UPTK_C_32I;
    else if (type_string.compare("UPTK_R_32U") == 0)
        return UPTK_R_32U;
    else if (type_string.compare("UPTK_C_32U") == 0)
        return UPTK_C_32U;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}
#endif