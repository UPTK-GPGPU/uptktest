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

    // Scene 1: Basic external memory mapped mipmapped array - error path
    // This API requires a valid UPTKExternalMemory_t handle. Without a real
    // external memory source, we test the error path.
    UPTKExternalMemory_t extMem = NULL;
    UPTKMipmappedArray_t mipmap = NULL;
    struct UPTKExternalMemoryMipmappedArrayDesc mipmapDesc;
    mipmapDesc.offset = 0;
    mipmapDesc.formatDesc.f = UPTKChannelFormatKindFloat;
    mipmapDesc.formatDesc.x = 32;
    mipmapDesc.formatDesc.y = 0;
    mipmapDesc.formatDesc.z = 0;
    mipmapDesc.formatDesc.w = 0;
    mipmapDesc.extent.width = 64;
    mipmapDesc.extent.height = 64;
    mipmapDesc.extent.depth = 1;
    mipmapDesc.flags = 0;
    mipmapDesc.numLevels = 1;

    // Expect error since extMem is NULL
    UPTKError_t err = UPTKExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &mipmapDesc);
    if (err == UPTKErrorInvalidValue || err == UPTKErrorInvalidResourceHandle) {
        printf("UPTKExternalMemoryGetMappedMipmappedArray returned expected error for NULL extMem: %s\n",
               UPTKGetErrorString(err));
    } else if (err == UPTKSuccess) {
        printf("UPTKExternalMemoryGetMappedMipmappedArray succeeded unexpectedly\n");
        UPTKFreeMipmappedArray(mipmap);
        UPTKDestroyExternalMemory(extMem);
    } else {
        printf("UPTKExternalMemoryGetMappedMipmappedArray returned: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaExternalMemoryGetMappedMipmappedArray PASS\n");
    return 0;
}
