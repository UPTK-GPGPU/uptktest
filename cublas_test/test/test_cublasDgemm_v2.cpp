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
    double hA[16],hB[16],hC[16];
    for(int i=0;i<16;i++){hA[i]=(double)(i+1);hB[i]=(double)(i+1);hC[i]=0;}
    double ha=1.0,hb=0.0;
    double *dA=NULL,*dB=NULL,*dC=NULL;
    UPTKMalloc(&dA,16*sizeof(double));UPTKMalloc(&dB,16*sizeof(double));UPTKMalloc(&dC,16*sizeof(double));
    UPTKMemcpy(dA,hA,16*sizeof(double),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dB,hB,16*sizeof(double),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dC,hC,16*sizeof(double),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasDgemm(handle,UPTKBLAS_OP_N,UPTKBLAS_OP_N,m,n,k,&ha,dA,m,dB,m,&hb,dC,m));
    UPTKFree(dA);UPTKFree(dB);UPTKFree(dC);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasDgemm_v2 PASS\n");
    return 0;
}
