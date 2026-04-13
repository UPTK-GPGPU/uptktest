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

    int m=4, n=4;
    float hA[16];
    for(int i=0;i<16;i++) hA[i]=(float)(i%4+1);
    float hx[4]={1,2,3,4}, hy[4]={0};
    float ha=1.0f, hb=0.0f;
    float *dA=NULL, *dx=NULL, *dy=NULL;
    UPTKMalloc(&dA,16*sizeof(float));UPTKMalloc(&dx,4*sizeof(float));UPTKMalloc(&dy,4*sizeof(float));
    UPTKMemcpy(dA,hA,16*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dx,hx,4*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(float),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasSgemv(handle,UPTKBLAS_OP_N,m,n,&ha,dA,m,dx,1,&hb,dy,1));
    float hr[4];
    UPTKMemcpy(hr,dy,4*sizeof(float),UPTKMemcpyDeviceToHost);
    printf("  UPTKblasSgemv: %f %f %f %f\n",hr[0],hr[1],hr[2],hr[3]);
    CHECK_CUBLAS(UPTKblasSgemv(handle,UPTKBLAS_OP_T,m,n,&ha,dA,m,dx,1,&hb,dy,1));
    UPTKFree(dA);UPTKFree(dx);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSgemv_v2 PASS\n");
    return 0;
}
