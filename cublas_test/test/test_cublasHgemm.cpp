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

    int m=4,n=4,k=4;
    __half hA[16],hB[16],hC[16];
    for(int i=0;i<16;i++){hA[i]=__float2half((float)(i+1));hB[i]=__float2half((float)(i+1));hC[i]=__float2half(0);}
    __half ha=__float2half(1.0f),hb=__float2half(0.0f);
    __half *dA=NULL,*dB=NULL,*dC=NULL;
    UPTKMalloc(&dA,16*sizeof(__half));UPTKMalloc(&dB,16*sizeof(__half));UPTKMalloc(&dC,16*sizeof(__half));
    UPTKMemcpy(dA,hA,16*sizeof(__half),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dB,hB,16*sizeof(__half),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dC,hC,16*sizeof(__half),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasHgemm(handle,UPTKBLAS_OP_N,UPTKBLAS_OP_N,m,n,k,&ha,dA,m,dB,m,&hb,dC,m));
    UPTKFree(dA);UPTKFree(dB);UPTKFree(dC);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasHgemm PASS\n");
    return 0;
}
