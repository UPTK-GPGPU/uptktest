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

    // Scene 1: Create and free a mipmapped array
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKExtent extent = make_UPTKExtent(64, 64, 1);
    UPTKMipmappedArray_t mipmap = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipmap, &channelDesc, extent, 1, 0));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipmap));
    printf("Basic mipmapped array malloc/free succeeded\n");

    // Scene 2: Create and free a mipmapped array with multiple levels
    UPTKMipmappedArray_t mipmap2 = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipmap2, &channelDesc, extent, 1, 3));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipmap2));
    printf("Mipmapped array with multiple levels malloc/free succeeded\n");

    // Scene 3: Free NULL mipmapped array (returns error on some implementations)
    UPTKError_t err = UPTKFreeMipmappedArray(NULL);
    if (err == UPTKSuccess) {
        printf("Free NULL mipmapped array succeeded\n");
    } else if (err == UPTKErrorInvalidValue) {
        printf("Free NULL mipmapped array returned UPTKErrorInvalidValue (expected)\n");
    } else {
        printf("CUDA error: unexpected error for NULL mipmapped array: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaFreeMipmappedArray PASS\n");
    return 0;
}
