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

    int n=4;
    float ha=2.0f,hx[4]={1,2,3,4};
    float* dx=NULL;
    UPTKMalloc(&dx,n*sizeof(float));
    UPTKMemcpy(dx,hx,n*sizeof(float),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasSscal(handle,n,&ha,dx,1));
    float hr[4];
    UPTKMemcpy(hr,dx,n*sizeof(float),UPTKMemcpyDeviceToHost);
    printf("  sscal: %f %f %f %f\n",hr[0],hr[1],hr[2],hr[3]);
    UPTKFree(dx);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSscal_v2 PASS\n");
    return 0;
}
