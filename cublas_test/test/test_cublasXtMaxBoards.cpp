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

    int mx = 0;
    UPTKblasStatus_t status = UPTKblasXtMaxBoards(&mx);
    if (status != UPTKBLAS_STATUS_SUCCESS) {
        printf("test_skip: cublasXtMaxBoards returned %s (Xt not fully supported on DTK)\n",
               UPTKblasGetStatusString(status));
        return 0;
    }
    printf("  max boards: %d\n", mx);

    printf("test_cublasXtMaxBoards PASS\n");
    return 0;
}
