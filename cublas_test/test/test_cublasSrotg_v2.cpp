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

    float ha=3.0f,hb=4.0f,hc,hs;
    CHECK_CUBLAS(UPTKblasSrotg(handle,&ha,&hb,&hc,&hs));
    printf("  srotg: a=%f b=%f c=%f s=%f\n",ha,hb,hc,hs);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSrotg_v2 PASS\n");
    return 0;
}
