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

    // Scene 1: cudaMemcpy2DArrayToArray_ptds (same function as non-ptds)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    int width = 32, height = 32;
    
    UPTKArray_t srcArray = NULL, dstArray = NULL;
    CHECK_CUDA(UPTKMallocArray(&srcArray, &channelDesc, width, height));
    CHECK_CUDA(UPTKMallocArray(&dstArray, &channelDesc, width, height));
    
    float *h_data = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) h_data[i] = (float)i;
    CHECK_CUDA(UPTKMemcpy2DToArray(srcArray, 0, 0, h_data, width * sizeof(float),
        width * sizeof(float), height, UPTKMemcpyHostToDevice));
    
    CHECK_CUDA(UPTKMemcpy2DArrayToArray(dstArray, 0, 0, srcArray, 0, 0,
        width * sizeof(float), height, UPTKMemcpyDeviceToDevice));
    
    CHECK_CUDA(UPTKFreeArray(srcArray));
    CHECK_CUDA(UPTKFreeArray(dstArray));
    free(h_data);

    // Scene 2: With offsets
    UPTKArray_t srcArray2 = NULL, dstArray2 = NULL;
    CHECK_CUDA(UPTKMallocArray(&srcArray2, &channelDesc, 64, 64));
    CHECK_CUDA(UPTKMallocArray(&dstArray2, &channelDesc, 64, 64));
    CHECK_CUDA(UPTKMemcpy2DArrayToArray(dstArray2, 8, 8, srcArray2, 4, 4,
        16 * sizeof(float), 16, UPTKMemcpyDeviceToDevice));
    CHECK_CUDA(UPTKFreeArray(srcArray2));
    CHECK_CUDA(UPTKFreeArray(dstArray2));

    printf("test_cudaMemcpy2DArrayToArray_ptds PASS\n");
    return 0;
}
