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
    float hAP[10];
    for(int i=0;i<10;i++) hAP[i]=(float)(i+1);
    float hA[16]={0};
    float *dAP=NULL,*dA=NULL;
    UPTKMalloc(&dAP,10*sizeof(float));UPTKMalloc(&dA,16*sizeof(float));
    UPTKMemcpy(dAP,hAP,10*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemset(dA,0,16*sizeof(float));
    CHECK_CUBLAS(UPTKblasStpttr(handle,UPTKBLAS_FILL_MODE_LOWER,n,dAP,dA,n));
    UPTKFree(dAP);UPTKFree(dA);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasStpttr PASS\n");
    return 0;
}
