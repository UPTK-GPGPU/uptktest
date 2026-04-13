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

    // Scene 1: cudaMemcpy2D_ptds (same function as non-ptds version)
    int width = 64, height = 64;
    size_t spitch = width * sizeof(float);
    size_t dpitch = 0;
    
    float *h_src = (float*)malloc(spitch * height);
    float *h_dst = (float*)malloc(spitch * height);
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr, &dpitch, width * sizeof(float), height));
    
    for (int i = 0; i < width * height; i++) h_src[i] = (float)i;
    
    CHECK_CUDA(UPTKMemcpy2D(d_ptr, dpitch, h_src, spitch,
        width * sizeof(float), height, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKMemcpy2D(h_dst, spitch, d_ptr, dpitch,
        width * sizeof(float), height, UPTKMemcpyDeviceToHost));
    
    int match = 1;
    for (int i = 0; i < width * height; i++) {
        if (h_src[i] != h_dst[i]) { match = 0; break; }
    }
    if (!match) { printf("Data mismatch\n"); return 1; }
    
    CHECK_CUDA(UPTKFree(d_ptr));
    free(h_src);
    free(h_dst);

    // Scene 2: cudaMemcpy2D_ptds with smaller region
    int w2 = 32, h2 = 32;
    float *h_src2 = (float*)malloc(w2 * h2 * sizeof(float));
    float *h_dst2 = (float*)malloc(w2 * h2 * sizeof(float));
    float *d_ptr2 = NULL;
    size_t dpitch2 = 0;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr2, &dpitch2, w2 * sizeof(float), h2));
    for (int i = 0; i < w2 * h2; i++) h_src2[i] = (float)i;
    CHECK_CUDA(UPTKMemcpy2D(d_ptr2, dpitch2, h_src2, w2 * sizeof(float),
        w2 * sizeof(float), h2, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKMemcpy2D(h_dst2, w2 * sizeof(float), d_ptr2, dpitch2,
        w2 * sizeof(float), h2, UPTKMemcpyDeviceToHost));
    CHECK_CUDA(UPTKFree(d_ptr2));
    free(h_src2);
    free(h_dst2);

    printf("test_cudaMemcpy2D_ptds PASS\n");
    return 0;
}
