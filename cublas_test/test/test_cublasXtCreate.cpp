#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_blas.h>
#include <UPTK_blasXt.h>
#include <cublasXt.h>
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

    // Scene 1: Create Xt handle
    UPTKblasXtHandle_t handle = NULL;
    CHECK_CUBLAS(UPTKblasXtCreate(&handle));
    if (handle == NULL) {
        printf("cuBLAS error: cublasXtCreate returned NULL handle\n");
        return 1;
    }

    // Scene 2: Select device and destroy
    int deviceId = 0;
    CHECK_CUBLAS(UPTKblasXtDeviceSelect(handle, 1, &deviceId));

    CHECK_CUBLAS(UPTKblasXtDestroy(handle));

    printf("test_cublasXtCreate PASS\n");
    return 0;
}
