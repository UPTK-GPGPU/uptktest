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
    cuComplex hx[4];
    for(int i=0;i<4;i++){hx[i].x=(float)(i+1);hx[i].y=(float)i;}
    float ha=1.0f;
    cuComplex *dx=NULL,*dAP=NULL;
    UPTKMalloc(&dx,4*sizeof(cuComplex));UPTKMalloc(&dAP,10*sizeof(cuComplex));
    UPTKMemcpy(dx,hx,4*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    UPTKMemset(dAP,0,10*sizeof(cuComplex));
    CHECK_CUBLAS(UPTKblasChpr(handle,UPTKBLAS_FILL_MODE_LOWER,n,&ha,dx,1,dAP));
    UPTKFree(dx);UPTKFree(dAP);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasChpr_v2 PASS\n");
    return 0;
}
