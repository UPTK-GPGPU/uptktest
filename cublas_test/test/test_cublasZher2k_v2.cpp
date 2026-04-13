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

    int n=4,k=4;
    cuDoubleComplex hA[16],hB[16],hC[16];
    for(int i=0;i<16;i++){hA[i].x=(double)(i+1);hA[i].y=0;hB[i].x=(double)(i+5);hB[i].y=0;hC[i].x=0;hC[i].y=0;}
    cuDoubleComplex ha={1,0};
    double hb=0.0;
    cuDoubleComplex *dA=NULL,*dB=NULL,*dC=NULL;
    UPTKMalloc(&dA,16*sizeof(cuDoubleComplex));UPTKMalloc(&dB,16*sizeof(cuDoubleComplex));UPTKMalloc(&dC,16*sizeof(cuDoubleComplex));
    UPTKMemcpy(dA,hA,16*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dB,hB,16*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dC,hC,16*sizeof(cuDoubleComplex),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasZher2k(handle,UPTKBLAS_FILL_MODE_LOWER,UPTKBLAS_OP_N,n,k,&ha,dA,n,dB,n,&hb,dC,n));
    UPTKFree(dA);UPTKFree(dB);UPTKFree(dC);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasZher2k_v2 PASS\n");
    return 0;
}
