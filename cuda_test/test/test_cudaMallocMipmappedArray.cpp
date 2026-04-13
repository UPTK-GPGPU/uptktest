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

    // Scene 1: Basic UPTKMallocMipmappedArray with single channel float
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKExtent extent = make_UPTKExtent(32, 32, 1);
    UPTKMipmappedArray_t mipArray = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipArray, &channelDesc, extent, 3, 0));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipArray));

    // Scene 2: UPTKMallocMipmappedArray with more levels
    UPTKMipmappedArray_t mipArray2 = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipArray2, &channelDesc, extent, 5, 0));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipArray2));

    // Scene 3: UPTKMallocMipmappedArray with int4 channel format
    UPTKChannelFormatDesc desc_int4 = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindSigned);
    UPTKExtent extent2 = make_UPTKExtent(16, 16, 1);
    UPTKMipmappedArray_t mipArray3 = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipArray3, &desc_int4, extent2, 2, 0));
    CHECK_CUDA(UPTKFreeMipmappedArray(mipArray3));

    printf("test_cudaMallocMipmappedArray PASS\n");
    return 0;
}
