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

    // Scene 1: Basic UPTKMemcpy2DToArrayAsync
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    int width = 64, height = 64;
    
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, width, height));
    
    float *h_src = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) h_src[i] = (float)i;
    
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    
    CHECK_CUDA(UPTKMemcpy2DToArrayAsync(array, 0, 0, h_src, width * sizeof(float),
        width * sizeof(float), height, UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    
    float *h_dst = (float*)malloc(width * height * sizeof(float));
    CHECK_CUDA(UPTKMemcpy2DFromArray(h_dst, width * sizeof(float), array, 0, 0,
        width * sizeof(float), height, UPTKMemcpyDeviceToHost));
    
    CHECK_CUDA(UPTKStreamDestroy(stream));
    CHECK_CUDA(UPTKFreeArray(array));
    free(h_src);
    free(h_dst);

    // Scene 2: UPTKMemcpy2DToArrayAsync with default stream
    UPTKArray_t array2 = NULL;
    CHECK_CUDA(UPTKMallocArray(&array2, &channelDesc, 32, 32));
    float *h_data2 = (float*)malloc(32 * 32 * sizeof(float));
    for (int i = 0; i < 32 * 32; i++) h_data2[i] = (float)i;
    CHECK_CUDA(UPTKMemcpy2DToArrayAsync(array2, 0, 0, h_data2, 32 * sizeof(float),
        32 * sizeof(float), 32, UPTKMemcpyHostToDevice, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFreeArray(array2));
    free(h_data2);

    printf("test_cudaMemcpy2DToArrayAsync PASS\n");
    return 0;
}
