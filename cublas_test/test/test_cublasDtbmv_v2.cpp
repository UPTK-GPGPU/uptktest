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

    int n=4,k=1,lda=4;
    double hA[16],hy[4];
    for(int i=0;i<16;i++) hA[i]=(double)(i+1);
    for(int i=0;i<4;i++) hy[i]=0;
    double *dA=NULL,*dy=NULL;
    UPTKMalloc(&dA,16*sizeof(double));UPTKMalloc(&dy,4*sizeof(double));
    UPTKMemcpy(dA,hA,16*sizeof(double),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,4*sizeof(double),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasDtbmv(handle,UPTKBLAS_FILL_MODE_LOWER,UPTKBLAS_OP_N,UPTKBLAS_DIAG_NON_UNIT,n,k,dA,lda,dy,1));
    UPTKFree(dA);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasDtbmv_v2 PASS\n");
    return 0;
}
