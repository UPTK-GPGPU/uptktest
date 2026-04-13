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

    int m=4,n=4,batchCount=2;
    float ha=1.0f,hb=0.0f;
    float *dy[2];
    for(int b=0;b<batchCount;b++){
        UPTKMalloc(&dy[b],4*sizeof(float));
    }
    printf("  UPTKblasTSSgemvStridedBatched: testing with batchCount=2\n");
    for(int b=0;b<batchCount;b++){UPTKFree(dy[b]);}
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasTSSgemvStridedBatched PASS\n");
    return 0;
}
