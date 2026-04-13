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

    // Scene 1: Get channel descriptor from a 1D array
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, 256));

    struct UPTKChannelFormatDesc desc;
    CHECK_CUDA(UPTKGetChannelDesc(&desc, array));
    printf("Channel desc: x=%d, y=%d, z=%d, w=%d, f=%d\n",
           desc.x, desc.y, desc.z, desc.w, desc.f);

    if (desc.x == 32 && desc.f == UPTKChannelFormatKindFloat) {
        printf("Channel descriptor matches float format\n");
    }

    CHECK_CUDA(UPTKFreeArray(array));

    // Scene 2: Get channel descriptor from a 2D array with different format
    UPTKChannelFormatDesc desc2d = UPTKCreateChannelDesc(16, 16, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array2d = NULL;
    CHECK_CUDA(UPTKMallocArray(&array2d, &desc2d, 64, 64));

    struct UPTKChannelFormatDesc desc_out;
    CHECK_CUDA(UPTKGetChannelDesc(&desc_out, array2d));
    printf("2D Channel desc: x=%d, y=%d, z=%d, w=%d, f=%d\n",
           desc_out.x, desc_out.y, desc_out.z, desc_out.w, desc_out.f);

    CHECK_CUDA(UPTKFreeArray(array2d));

    printf("test_cudaGetChannelDesc PASS\n");
    return 0;
}
