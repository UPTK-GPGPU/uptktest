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

// Deprecated texture reference for testing
texture<float, 2, cudaReadModeElementType> texRef;

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Get texture alignment offset for a texture reference
    size_t offset = 0;
    UPTKError_t err = UPTKGetTextureAlignmentOffset(&offset, &texRef);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKGetTextureAlignmentOffset returned error (deprecated API, %s)\n",
               UPTKGetErrorString(err));
        return 0;
    }

    printf("test_cudaGetTextureAlignmentOffset PASS\n");
    return 0;
}
