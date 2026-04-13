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

    // Scene 1: Create and free a 1D UPTKArray
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, 256));
    CHECK_CUDA(UPTKFreeArray(array));
    printf("1D array malloc/free succeeded\n");

    // Scene 2: Create and free a 2D UPTKArray
    UPTKArray_t array2d = NULL;
    CHECK_CUDA(UPTKMallocArray(&array2d, &channelDesc, 64, 64));
    CHECK_CUDA(UPTKFreeArray(array2d));
    printf("2D array malloc/free succeeded\n");

    // Scene 3: Free NULL array (should be safe)
    CHECK_CUDA(UPTKFreeArray(NULL));
    printf("Free NULL array succeeded\n");

    printf("test_cudaFreeArray PASS\n");
    return 0;
}
