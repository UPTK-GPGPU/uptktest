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

    int v = 0;
    UPTKblasStatus_t status = UPTKblasGetProperty(MAJOR_VERSION, &v);
    if (status == UPTKBLAS_STATUS_SUCCESS) {
        printf("  cuBLAS major version: %d\n", v);
    }
    status = UPTKblasGetProperty(MINOR_VERSION, &v);
    if (status == UPTKBLAS_STATUS_SUCCESS) {
        printf("  cuBLAS minor version: %d\n", v);
    }

    int ver = 0;
    CHECK_CUBLAS(UPTKblasGetVersion(handle, &ver));
    printf("  UPTKblasGetVersion: %d\n", ver);

    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasGetProperty PASS\n");
    return 0;
}
