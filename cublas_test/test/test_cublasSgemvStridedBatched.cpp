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

    int m=4,n=4,batchCount=2;
    float hA[32],hx[8],hy[8];
    for(int i=0;i<32;i++) hA[i]=(float)(i%4+1);
    for(int i=0;i<8;i++) hx[i]=(float)(i%4+1);
    for(int i=0;i<8;i++) hy[i]=0;
    float ha=1.0f,hb=0.0f;
    float *dA=NULL,*dx=NULL,*dy=NULL;
    UPTKMalloc(&dA,32*sizeof(float));UPTKMalloc(&dx,8*sizeof(float));UPTKMalloc(&dy,8*sizeof(float));
    UPTKMemcpy(dA,hA,32*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dx,hx,8*sizeof(float),UPTKMemcpyHostToDevice);
    UPTKMemcpy(dy,hy,8*sizeof(float),UPTKMemcpyHostToDevice);
    CHECK_CUBLAS(UPTKblasSgemvStridedBatched(handle,UPTKBLAS_OP_N,m,n,&ha,dA,m,16,dx,1,4,&hb,dy,1,4,batchCount));
    UPTKFree(dA);UPTKFree(dx);UPTKFree(dy);
    CHECK_CUBLAS(UPTKblasDestroy(handle));
    printf("test_cublasSgemvStridedBatched PASS\n");
    return 0;
}
