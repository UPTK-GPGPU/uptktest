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

    int r=4, c=4;
    float hA[16];
    for(int i=0;i<16;i++) hA[i]=(float)i;
    float* dB = NULL;
    UPTKMalloc(&dB, 16*sizeof(float));
    CHECK_CUBLAS(UPTKblasSetMatrix(r, c, sizeof(float), hA, r, dB, r));
    UPTKFree(dB);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSetMatrix PASS\n");
    return 0;
}
