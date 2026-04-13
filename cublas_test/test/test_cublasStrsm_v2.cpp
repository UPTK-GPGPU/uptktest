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
    float hA[16],hB[16];
    for(int i=0;i<16;i++){hA[i]=(float)(i+1);hB[i]=(float)(i+1);}
    float ha=1.0f;
    float *dA=NULL,*dB=NULL;
    UPTKMalloc(&dA,16*sizeof(float));UPTKMalloc(&dB,16*sizeof(float));
    UPTKMemcpy(dA,hA,16*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dB,hB,16*sizeof(float),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasStrsm(handle,UPTKBLAS_SIDE_LEFT,UPTKBLAS_FILL_MODE_LOWER,UPTKBLAS_OP_N,UPTKBLAS_DIAG_NON_UNIT,m,n,&ha,dA,m,dB,m));
    UPTKFree(dA);UPTKFree(dB);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasStrsm_v2 PASS\n");
    return 0;
}
