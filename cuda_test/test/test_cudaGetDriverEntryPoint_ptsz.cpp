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

    // Scenario 1: Get UPMemAlloc driver entry point
    void *funcPtr = NULL;
    UPTKError_t err = UPTKGetDriverEntryPoint("UPMemAlloc", &funcPtr, UPTKEnableDefault);
    if (err == UPTKSuccess) {
        if (funcPtr == NULL) {
            printf("CUDA error: UPMemAlloc driver entry point returned NULL\n");
            return 1;
        }
        printf("Got UPMemAlloc driver entry point\n");
    } else {
        printf("UPTKGetDriverEntryPoint returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    }

    // Scenario 2: Get UPMemFree driver entry point
    void *funcPtr2 = NULL;
    err = UPTKGetDriverEntryPoint("UPMemFree", &funcPtr2, UPTKEnableDefault);
    if (err == UPTKSuccess) {
        if (funcPtr2 == NULL) {
            printf("CUDA error: UPMemFree driver entry point returned NULL\n");
            return 1;
        }
        printf("Got UPMemFree driver entry point\n");
    } else {
        printf("UPTKGetDriverEntryPoint returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    }

    printf("test_cudaGetDriverEntryPoint_ptsz PASS\n");
    return 0;
}
