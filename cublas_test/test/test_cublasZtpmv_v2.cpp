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
    cuDoubleComplex hAP[10],hy[4];
    for(int i=0;i<10;i++){hAP[i].x=(float)(i+1);hAP[i].y=0;}
    for(int i=0;i<4;i++){hy[i].x=0;hy[i].y=0;}
    cuDoubleComplex *dAP=NULL,*dy=NULL;
    UPTKMalloc(&dAP,10*sizeof(cuDoubleComplex));UPTKMalloc(&dy,4*sizeof(cuDoubleComplex));
    UPTKMemcpy(dAP,hAP,10*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasZtpmv(handle,UPTKBLAS_FILL_MODE_LOWER,UPTKBLAS_OP_N,UPTKBLAS_DIAG_NON_UNIT,n,dAP,dy,1));
    UPTKFree(dAP);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasZtpmv_v2 PASS\n");
    return 0;
}
