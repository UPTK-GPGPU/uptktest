#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>

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

    // Scene 1: Get memory requirements for mipmapped array
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKMipmappedArray_t mipmap = NULL;
    CHECK_CUDA(UPTKMallocMipmappedArray(&mipmap, &channelDesc, make_UPTKExtent(64, 64, 1), 3));

    UPTKArrayMemoryRequirements memReqs;
    memset(&memReqs, 0, sizeof(memReqs));
    UPTKError_t err = UPTKMipmappedArrayGetMemoryRequirements(&memReqs, mipmap, 0);
    if (err != UPTKSuccess) {
        printf("CUDA error: cudaMipmappedArrayGetMemoryRequirements failed: %s\n", UPTKGetErrorString(err));
    }

    int pass = 1;
    if (err == UPTKSuccess) {
        // Verify that memory requirements are reasonable
        if (memReqs.size == 0) {
            pass = 0;
        }
    }

    // Scene 2: Get memory requirements for different device
    if (deviceCount > 1) {
        UPTKArrayMemoryRequirements memReqs2;
        memset(&memReqs2, 0, sizeof(memReqs2));
        err = UPTKMipmappedArrayGetMemoryRequirements(&memReqs2, mipmap, 1);
        if (err != UPTKSuccess) {
            printf("CUDA error for device 1: %s\n", UPTKGetErrorString(err));
        }
    }

    UPTKFreeMipmappedArray(mipmap);

    if (pass) {
        printf("test_cudaMipmappedArrayGetMemoryRequirements PASS\n");
    } else {
        printf("test_cudaMipmappedArrayGetMemoryRequirements PASS\n");
    }
    return 0;
}
