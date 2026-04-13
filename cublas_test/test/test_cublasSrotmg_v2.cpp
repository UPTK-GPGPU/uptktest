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

    float hd1=1.0f,hd2=1.0f,hx1=3.0f;
    const float hy1=4.0f;
    float hp[5];
    CHECK_CUBLAS(UPTKblasSrotmg(handle,&hd1,&hd2,&hx1,&hy1,hp));
    printf("  srotmg: d1=%f d2=%f x1=%f\n",hd1,hd2,hx1);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSrotmg_v2 PASS\n");
    return 0;
}
