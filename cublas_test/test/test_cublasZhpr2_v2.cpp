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
    cuDoubleComplex hx[4],hy[4];
    for(int i=0;i<4;i++){hx[i].x=(double)(i+1);hx[i].y=0;hy[i].x=(double)(i+5);hy[i].y=0;}
    cuDoubleComplex ha=make_cuDoubleComplex(1.0,0.0);
    cuDoubleComplex *dx=NULL,*dy=NULL,*dAP=NULL;
    UPTKMalloc(&dx,4*sizeof(cuDoubleComplex));UPTKMalloc(&dy,4*sizeof(cuDoubleComplex));UPTKMalloc(&dAP,10*sizeof(cuDoubleComplex));
    UPTKMemcpy(dx,hx,4*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemset(dAP,0,10*sizeof(cuDoubleComplex));
    CHECK_CUBLAS(UPTKblasZhpr2(handle,UPTKBLAS_FILL_MODE_LOWER,n,&ha,dx,1,dy,1,dAP));
    UPTKFree(dx);UPTKFree(dy);UPTKFree(dAP);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasZhpr2_v2 PASS\n");
    return 0;
}
