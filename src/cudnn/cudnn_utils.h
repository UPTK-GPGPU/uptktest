#ifndef CUDNN_UTILS_H
#define CUDNN_UTILS_H

#include <gtest/gtest.h>

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cudnn API error checking
#define CUDNN_CHECK(err)                                                                        \
    do {                                                                                           \
        cudnnStatus_t err_ = (err);                                                             \
        if (err_ != CUDNN_STATUS_SUCCESS) {                                                     \
            printf("cudnn error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cudnn error");                                            \
        }                                                                                          \
    } while (0)


#endif
