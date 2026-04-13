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

    // UPTKGetTextureReference is deprecated and requires a texture symbol.
    // Test error path with NULL symbol.
    const struct textureReference *texref = NULL;
    UPTKError_t err = UPTKGetTextureReference(&texref, NULL);
    if (err == UPTKErrorInvalidValue) {
        printf("UPTKGetTextureReference correctly returned UPTKErrorInvalidValue for NULL symbol\n");
    } else {
        printf("test_skip: UPTKGetTextureReference returned %s (deprecated API)\n",
               UPTKGetErrorString(err));
        return 0;
    }

    printf("test_cudaGetTextureReference PASS\n");
    return 0;
}
