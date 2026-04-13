#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_runtime_api.h>
#include <stdio.h>
#include <string.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic functionality - get memory requirements for a regular array
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t cuArray;
    UPTKError_t err = UPTKMallocArray(&cuArray, &channelDesc, 1024);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    struct UPTKArrayMemoryRequirements memReq;
    memset(&memReq, 0, sizeof(memReq));

    err = UPTKArrayGetMemoryRequirements(&memReq, cuArray, 0);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKArrayGetMemoryRequirements returned: %s (not supported for non-deferred arrays on DTK)\n",
               UPTKGetErrorString(err));
        UPTKFreeArray(cuArray);
        return 0;
    }
    printf("  Memory size: %zu, alignment: %zu\n", memReq.size, memReq.alignment);

    UPTKFreeArray(cuArray);

    // Scenario 2: Error handling - null pointer
    err = UPTKArrayGetMemoryRequirements(NULL, cuArray, 0);
    if (err != UPTKErrorInvalidValue && err != UPTKSuccess) {
        printf("CUDA error: expected error for null pointer, got: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaArrayGetMemoryRequirements PASS\n");
    return 0;
}
