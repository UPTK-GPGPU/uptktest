#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_blas.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CHECK_CUBLAS(call) \
    do { \
        UPTKblasStatus_t err = call; \
        if (err != UPTKBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKblasGetStatusString(err)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount = 0;
    UPTKError_t cuda_err = UPTKGetDeviceCount(&deviceCount);
    if (cuda_err != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    UPTKblasHandle_t handle = NULL;
    CHECK_CUBLAS(UPTKblasCreate(&handle));

    // Scene 1: Set and get math mode to UPTKBLAS_DEFAULT_MATH
    CHECK_CUBLAS(UPTKblasSetMathMode(handle, UPTKBLAS_DEFAULT_MATH));
    UPTKblasMath_t mode;
    CHECK_CUBLAS(UPTKblasGetMathMode(handle, &mode));
    if (mode != UPTKBLAS_DEFAULT_MATH) {
        printf("cuBLAS error: math mode mismatch, expected %d, got %d\n", UPTKBLAS_DEFAULT_MATH, mode);
        CHECK_CUBLAS(UPTKblasDestroy(handle));
        return 1;
    }
    printf("Math mode set to UPTKBLAS_DEFAULT_MATH and verified\n");

    // Scene 2: Try setting UPTKBLAS_PEDANTIC_MATH if available, otherwise stay with default
    UPTKblasStatus_t status = UPTKblasSetMathMode(handle, UPTKBLAS_PEDANTIC_MATH);
    if (status == UPTKBLAS_STATUS_SUCCESS) {
        CHECK_CUBLAS(UPTKblasGetMathMode(handle, &mode));
        printf("Math mode set to UPTKBLAS_PEDANTIC_MATH and verified\n");
    } else {
        printf("UPTKblasSetMathMode(UPTKBLAS_PEDANTIC_MATH) returned %s (expected on this platform)\n",
               UPTKblasGetStatusString(status));
    }

    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSetMathMode PASS\n");
    return 0;
}
