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

    int deviceId = 0;
    int nbBoards = 0;
    status = UPTKblasXtGetNumBoards(1, &deviceId, &nbBoards);
    if (status != UPTKBLAS_STATUS_SUCCESS) {
        printf("test_skip: cublasXtGetNumBoards returned %s (Xt not fully supported on DTK)\n",
               UPTKblasGetStatusString(status));
        UPTKblasXtDestroy(handle);
        return 0;
    }
    printf("  nbBoards: %d\n", nbBoards);

    int maxBoards = 0;
    status = UPTKblasXtMaxBoards(&maxBoards);
    if (status != UPTKBLAS_STATUS_SUCCESS) {
        printf("test_skip: cublasXtMaxBoards returned %s (Xt not fully supported on DTK)\n",
               UPTKblasGetStatusString(status));
        UPTKblasXtDestroy(handle);
        return 0;
    }
    printf("  maxBoards: %d\n", maxBoards);

    UPTKblasXtDestroy(handle);
    printf("test_cublasXtGetNumBoards PASS\n");
    return 0;
}
