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

    // Scene 1: Basic UPTKMallocArray with 1D channel format
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, 256, 0));
    CHECK_CUDA(UPTKFreeArray(array));

    // Scene 2: UPTKMallocArray with 2D dimensions
    UPTKArray_t array2d = NULL;
    CHECK_CUDA(UPTKMallocArray(&array2d, &channelDesc, 128, 128));
    CHECK_CUDA(UPTKFreeArray(array2d));

    // Scene 3: UPTKMallocArray with UPTKArraySurfaceLoadStore flag
    UPTKArray_t array_surf = NULL;
    CHECK_CUDA(UPTKMallocArray(&array_surf, &channelDesc, 64, 64, UPTKArraySurfaceLoadStore));
    CHECK_CUDA(UPTKFreeArray(array_surf));

    // Scene 4: UPTKMallocArray with int2 channel format
    UPTKChannelFormatDesc desc_int2 = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindSigned);
    UPTKArray_t array_int2 = NULL;
    CHECK_CUDA(UPTKMallocArray(&array_int2, &desc_int2, 32, 32));
    CHECK_CUDA(UPTKFreeArray(array_int2));

    printf("test_cudaMallocArray PASS\n");
    return 0;
}
