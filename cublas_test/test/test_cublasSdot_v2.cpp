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
    float hx[4]={1,2,3,4},hy[4]={5,6,7,8};
    float *dx=NULL,*dy=NULL;
    UPTKMalloc(&dx,n*sizeof(float));UPTKMalloc(&dy,n*sizeof(float));
    UPTKMemcpy(dx,hx,n*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,n*sizeof(float),UPTKMemcpyHostToDevice);
    float r=0;
    CHECK_CUBLAS(UPTKblasSdot(handle,n,dx,1,dy,1,&r));
    printf("  sdot: %f\n",r);
    UPTKFree(dx);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSdot_v2 PASS\n");
    return 0;
}
