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
    cuDoubleComplex hAP[10];
    for(int i=0;i<10;i++){hAP[i].x=(double)(i+1);hAP[i].y=0;}
    cuDoubleComplex hA[16];
    for(int i=0;i<16;i++){hA[i].x=0;hA[i].y=0;}
    cuDoubleComplex *dAP=NULL,*dA=NULL;
    UPTKMalloc(&dAP,10*sizeof(cuDoubleComplex));UPTKMalloc(&dA,16*sizeof(cuDoubleComplex));
    UPTKMemcpy(dAP,hAP,10*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemset(dA,0,16*sizeof(cuDoubleComplex));
    CHECK_CUBLAS(UPTKblasZtpttr(handle,UPTKBLAS_FILL_MODE_LOWER,n,dAP,dA,n));
    UPTKFree(dAP);UPTKFree(dA);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasZtpttr PASS\n");
    return 0;
}
