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

    int m=4,n=4;
    cuDoubleComplex hx[4],hy[4];
    for(int i=0;i<4;i++){hx[i].x=(float)(i+1);hx[i].y=0;hy[i].x=(float)(i+5);hy[i].y=0;}
    cuDoubleComplex ha={1,0};
    cuDoubleComplex *dx=NULL,*dy=NULL,*dA=NULL;
    UPTKMalloc(&dx,4*sizeof(cuDoubleComplex));UPTKMalloc(&dy,4*sizeof(cuDoubleComplex));UPTKMalloc(&dA,16*sizeof(cuDoubleComplex));
    UPTKMemcpy(dx,hx,4*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemset(dA,0,16*sizeof(cuDoubleComplex));
    CHECK_CUBLAS(UPTKblasZgeru(handle,m,n,&ha,dx,1,dy,1,dA,m));
    UPTKFree(dx);UPTKFree(dy);UPTKFree(dA);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasZgeru_v2 PASS\n");
    return 0;
}
