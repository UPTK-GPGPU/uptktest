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

    int m=4,n=4,k=4,batchCount=2;
    float hA[32],hB[32],hC[32];
    for(int i=0;i<32;i++){hA[i]=(float)(i%4+1);hB[i]=(float)(i%4+1);hC[i]=0;}
    float ha=1.0f,hb=0.0f;
    float *dA=NULL,*dB=NULL,*dC=NULL;
    UPTKMalloc(&dA,32*sizeof(float));UPTKMalloc(&dB,32*sizeof(float));UPTKMalloc(&dC,32*sizeof(float));
    UPTKMemcpy(dA,hA,32*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dB,hB,32*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dC,hC,32*sizeof(float),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasGemmStridedBatchedEx(handle,UPTKBLAS_OP_N,UPTKBLAS_OP_N,m,n,k,&ha,dA,UPTK_R_32F,m,16,dB,UPTK_R_32F,m,16,&hb,dC,UPTK_R_32F,m,16,batchCount,UPTKBLAS_COMPUTE_32F,UPTKBLAS_GEMM_DEFAULT));
    UPTKFree(dA);UPTKFree(dB);UPTKFree(dC);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasGemmStridedBatchedEx PASS\n");
    return 0;
}
