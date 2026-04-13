#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_blas.h>
#include <UPTK_blasXt.h>
#include <cublasXt.h>
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

    UPTKblasXtHandle_t handle = NULL;

    CHECK_CUBLAS(UPTKblasXtCreate(&handle));
    int devices[1] = {0};
    CHECK_CUBLAS(UPTKblasXtDeviceSelect(handle, 1, devices));
    int n=4,k=4;
    cuDoubleComplex hA[16],hB[16],hC[16];
    for(int i=0;i<16;i++){hA[i].x=(double)(i+1);hA[i].y=0;hB[i].x=(double)(i+5);hB[i].y=0;hC[i].x=0;hC[i].y=0;}
    cuDoubleComplex ha={1,0};
    double hb=0.0;
    CHECK_CUBLAS(UPTKblasXtZher2k(handle,UPTKBLAS_FILL_MODE_LOWER,UPTKBLAS_OP_N,n,k,&ha,hA,n,hB,n,&hb,hC,n));
    CHECK_CUBLAS(UPTKblasXtDestroy(handle));
    printf("test_cublasXtZher2k PASS\n");
    return 0;
}
