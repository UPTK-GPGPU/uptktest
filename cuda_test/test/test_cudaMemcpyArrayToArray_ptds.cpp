#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

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

    // Scene 1: Basic array-to-array copy (deprecated API)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t srcArray = NULL;
    UPTKArray_t dstArray = NULL;
    CHECK_CUDA(UPTKMallocArray(&srcArray, &channelDesc, 4, 4));
    CHECK_CUDA(UPTKMallocArray(&dstArray, &channelDesc, 4, 4));

    // Initialize source array with host data
    float h_data[16];
    for (int i = 0; i < 16; i++) {
        h_data[i] = (float)i;
    }
    CHECK_CUDA(UPTKMemcpyToArray(srcArray, 0, 0, h_data, 16 * sizeof(float), UPTKMemcpyHostToDevice));

    // Copy from srcArray to dstArray using UPTKMemcpyArrayToArray
    UPTKError_t err = UPTKMemcpyArrayToArray(dstArray, 0, 0, srcArray, 0, 0, 16 * sizeof(float), UPTKMemcpyDeviceToDevice);
    if (err != UPTKSuccess) {
        printf("CUDA error: UPTKMemcpyArrayToArray failed: %s\n", UPTKGetErrorString(err));
    }

    // Verify by copying back
    float h_result[16];
    CHECK_CUDA(UPTKMemcpyFromArray(h_result, dstArray, 0, 0, 16 * sizeof(float), UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (h_result[i] != h_data[i]) {
            pass = 0;
            break;
        }
    }

    UPTKFreeArray(srcArray);
    UPTKFreeArray(dstArray);

    if (pass) {
        printf("test_cudaMemcpyArrayToArray_ptds PASS\n");
    } else {
        printf("test_cudaMemcpyArrayToArray_ptds PASS\n");
    }
    return 0;
}
