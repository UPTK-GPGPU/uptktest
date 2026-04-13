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
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Basic UPTKMalloc3DArray with float channel format
    UPTKArray_t array = NULL;
    struct UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    struct UPTKExtent extent = make_UPTKExtent(64, 64, 1);
    CHECK_CUDA(UPTKMalloc3DArray(&array, &channelDesc, extent));
    if (array == NULL) {
        printf("UPTKMalloc3DArray returned NULL\n");
        return 1;
    }
    printf("Scenario 1: UPTKMalloc3DArray float 64x64x1 PASS\n");
    CHECK_CUDA(UPTKFreeArray(array));

    // Scenario 2: UPTKMalloc3DArray with 3D extent and int channel
    channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindSigned);
    extent = make_UPTKExtent(32, 32, 32);
    CHECK_CUDA(UPTKMalloc3DArray(&array, &channelDesc, extent));
    printf("Scenario 2: UPTKMalloc3DArray int 32x32x32 PASS\n");
    CHECK_CUDA(UPTKFreeArray(array));

    // Scenario 3: UPTKMalloc3DArray with UPTKArraySurfaceLoadStore flag
    channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    extent = make_UPTKExtent(16, 16, 16);
    CHECK_CUDA(UPTKMalloc3DArray(&array, &channelDesc, extent, UPTKArraySurfaceLoadStore));
    printf("Scenario 3: UPTKMalloc3DArray with surface flag PASS\n");
    CHECK_CUDA(UPTKFreeArray(array));

    // Scenario 4: Error handling - NULL array pointer
    channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    extent = make_UPTKExtent(64, 64, 1);
    UPTKError_t err = UPTKMalloc3DArray(NULL, &channelDesc, extent);
    printf("Scenario 4: NULL array pointer returned: %s\n", UPTKGetErrorString(err));

    // Scenario 5: UPTKMalloc3DArray with zero extent
    extent = make_UPTKExtent(0, 0, 0);
    err = UPTKMalloc3DArray(&array, &channelDesc, extent);
    printf("Scenario 5: UPTKMalloc3DArray with zero extent returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaMalloc3DArray PASS\n");
    return 0;
}
