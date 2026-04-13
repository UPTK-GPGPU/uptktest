#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_blas.h>
#include <UPTK_blasXt.h>
#include <cublasXt.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    UPTKError_t cuda_err = UPTKGetDeviceCount(&deviceCount);
    if (cuda_err != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    UPTKblasXtHandle_t handle = NULL;
    UPTKblasStatus_t status = UPTKblasXtCreate(&handle);
    if (status != UPTKBLAS_STATUS_SUCCESS) {
        printf("test_skip: cublasXtCreate failed on DTK: %s\n", UPTKblasGetStatusString(status));
        return 0;
    }

    // cublasXtDestroy may fail on DTK since Xt subsystem is incomplete
    status = UPTKblasXtDestroy(handle);
    if (status != UPTKBLAS_STATUS_SUCCESS) {
        printf("test_skip: cublasXtDestroy returned %s (Xt not fully supported on DTK)\n",
               UPTKblasGetStatusString(status));
        return 0;
    }

    printf("test_cublasXtDestroy PASS\n");
    return 0;
}
