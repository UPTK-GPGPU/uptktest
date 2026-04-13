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
    float hx[4]={1,2,3,4};
    float ha=1.0f;
    float *dx=NULL,*dAP=NULL;
    UPTKMalloc(&dx,4*sizeof(float));UPTKMalloc(&dAP,10*sizeof(float));
    UPTKMemcpy(dx,hx,4*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemset(dAP,0,10*sizeof(float));
    CHECK_CUBLAS(UPTKblasSspr(handle,UPTKBLAS_FILL_MODE_LOWER,n,&ha,dx,1,dAP));
    UPTKFree(dx);UPTKFree(dAP);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSspr_v2 PASS\n");
    return 0;
}
