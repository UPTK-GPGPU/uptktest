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

    // Scene 1: UPTKMemcpy2DToArrayAsync with per-thread default stream (ptsz variant)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    int width = 64, height = 64;
    
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, width, height));
    
    float *h_src = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) h_src[i] = (float)i;
    
    CHECK_CUDA(UPTKMemcpy2DToArrayAsync(array, 0, 0, h_src, width * sizeof(float),
        width * sizeof(float), height, UPTKMemcpyHostToDevice, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());
    
    CHECK_CUDA(UPTKFreeArray(array));
    free(h_src);

    printf("test_cudaMemcpy2DToArrayAsync_ptsz PASS\n");
    return 0;
}
