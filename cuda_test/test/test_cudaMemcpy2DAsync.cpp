#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        UPTKError_t err = call; \
        if (err != UPTKSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic UPTKMemcpy2DAsync H2D and D2H
    int width = 64, height = 64;
    size_t spitch = width * sizeof(float);
    size_t dpitch = 0;
    
    float *h_src = (float*)malloc(spitch * height);
    float *h_dst = (float*)malloc(spitch * height);
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr, &dpitch, width * sizeof(float), height));
    
    for (int i = 0; i < width * height; i++) h_src[i] = (float)i;
    
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    
    CHECK_CUDA(UPTKMemcpy2DAsync(d_ptr, dpitch, h_src, spitch,
        width * sizeof(float), height, UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKMemcpy2DAsync(h_dst, spitch, d_ptr, dpitch,
        width * sizeof(float), height, UPTKMemcpyDeviceToHost, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    
    CHECK_CUDA(UPTKStreamDestroy(stream));
    CHECK_CUDA(UPTKFree(d_ptr));
    free(h_src);
    free(h_dst);

    // Scene 2: UPTKMemcpy2DAsync with default stream
    float *h_src2 = (float*)malloc(width * height * sizeof(float));
    float *h_dst2 = (float*)malloc(width * height * sizeof(float));
    float *d_ptr2 = NULL;
    size_t dpitch2 = 0;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr2, &dpitch2, width * sizeof(float), height));
    for (int i = 0; i < width * height; i++) h_src2[i] = (float)i;
    CHECK_CUDA(UPTKMemcpy2DAsync(d_ptr2, dpitch2, h_src2, width * sizeof(float),
        width * sizeof(float), height, UPTKMemcpyHostToDevice, 0));
    CHECK_CUDA(UPTKMemcpy2DAsync(h_dst2, width * sizeof(float), d_ptr2, dpitch2,
        width * sizeof(float), height, UPTKMemcpyDeviceToHost, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFree(d_ptr2));
    free(h_src2);
    free(h_dst2);

    printf("test_cudaMemcpy2DAsync PASS\n");
    return 0;
}
