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
    cuComplex hx[4],hy[4];
    for(int i=0;i<n;i++){hx[i].x=(float)(i+1);hx[i].y=(float)i;hy[i].x=(float)(i+5);hy[i].y=(float)(i+1);}
    cuComplex *dx=NULL,*dy=NULL;
    UPTKMalloc(&dx,n*sizeof(cuComplex));UPTKMalloc(&dy,n*sizeof(cuComplex));
    UPTKMemcpy(dx,hx,n*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,n*sizeof(cuComplex),UPTKMemcpyHostToDevice);
    cuComplex r;
    CHECK_CUBLAS(UPTKblasCdotc(handle,n,dx,1,dy,1,&r));
    printf("  cdotc: (%f,%f)\n",r.x,r.y);
    UPTKFree(dx);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasCdotc_v2 PASS\n");
    return 0;
}
