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

    // Scene 1: cudaMemcpy2DToArray_ptds (same function as non-ptds)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    int width = 64, height = 64;
    
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, width, height));
    
    float *h_src = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) h_src[i] = (float)i;
    
    CHECK_CUDA(UPTKMemcpy2DToArray(array, 0, 0, h_src, width * sizeof(float),
        width * sizeof(float), height, UPTKMemcpyHostToDevice));
    
    CHECK_CUDA(UPTKFreeArray(array));
    free(h_src);

    // Scene 2: With offsets
    UPTKArray_t array2 = NULL;
    CHECK_CUDA(UPTKMallocArray(&array2, &channelDesc, 128, 128));
    float *h_data2 = (float*)malloc(32 * 32 * sizeof(float));
    for (int i = 0; i < 32 * 32; i++) h_data2[i] = (float)i;
    CHECK_CUDA(UPTKMemcpy2DToArray(array2, 16, 16, h_data2, 32 * sizeof(float),
        32 * sizeof(float), 32, UPTKMemcpyHostToDevice));
    CHECK_CUDA(UPTKFreeArray(array2));
    free(h_data2);

    printf("test_cudaMemcpy2DToArray_ptds PASS\n");
    return 0;
}
