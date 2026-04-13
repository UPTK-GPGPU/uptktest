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
    cuComplex hA[16],hx[4],hy[4];
    for(int i=0;i<16;i++){hA[i].x=(float)(i+1);hA[i].y=0;}
    for(int i=0;i<4;i++){hx[i].x=(float)(i+1);hx[i].y=0;hy[i].x=0;hy[i].y=0;}
    cuComplex ha={1,0},hb={0,0};
    cuComplex *dA=NULL,*dx=NULL,*dy=NULL;
    UPTKMalloc(&dA,16*sizeof(cuComplex));UPTKMalloc(&dx,4*sizeof(cuComplex));UPTKMalloc(&dy,4*sizeof(cuComplex));
    UPTKMemcpy(dA,hA,16*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dx,hx,4*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasChemv(handle,UPTKBLAS_FILL_MODE_LOWER,n,&ha,dA,n,dx,1,&hb,dy,1));
    UPTKFree(dA);UPTKFree(dx);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasChemv_v2 PASS\n");
    return 0;
}
