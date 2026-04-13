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

    // Scenario 1: Create mipmapped array and get level 0
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKMipmappedArray_t mipmappedArray = NULL;
    UPTKExtent extent = make_UPTKExtent(64, 64, 1);
    unsigned int numLevels = 1;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipmappedArray, &channelDesc, extent, numLevels, 0));

    UPTKArray_t levelArray = NULL;
    CHECK_CUDA(UPTKGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0));
    if (levelArray == NULL) {
        printf("CUDA error: UPTKGetMipmappedArrayLevel returned NULL for level 0\n");
        CHECK_CUDA(UPTKFreeMipmappedArray(mipmappedArray));
        return 1;
    }

    // Scenario 2: Get level from a multi-level mipmapped array
    UPTKMipmappedArray_t mipmappedArray2 = NULL;
    UPTKExtent extent2 = make_UPTKExtent(128, 128, 1);
    unsigned int numLevels2 = 3;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipmappedArray2, &channelDesc, extent2, numLevels2, 0));

    UPTKArray_t levelArray2 = NULL;
    CHECK_CUDA(UPTKGetMipmappedArrayLevel(&levelArray2, mipmappedArray2, 1));
    if (levelArray2 == NULL) {
        printf("CUDA error: UPTKGetMipmappedArrayLevel returned NULL for level 1\n");
        CHECK_CUDA(UPTKFreeMipmappedArray(mipmappedArray));
        CHECK_CUDA(UPTKFreeMipmappedArray(mipmappedArray2));
        return 1;
    }

    CHECK_CUDA(UPTKFreeMipmappedArray(mipmappedArray));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipmappedArray2));

    printf("test_cudaGetMipmappedArrayLevel PASS\n");
    return 0;
}
