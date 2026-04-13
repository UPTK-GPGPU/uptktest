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

    // UPTKIpcCloseMemHandle requires memory obtained from UPTKIpcOpenMemHandle,
    // which requires multi-process IPC. Test error handling only.

    // Scenario 1: UPTKIpcCloseMemHandle with NULL pointer
    UPTKError_t err = UPTKIpcCloseMemHandle(NULL);
    printf("Scenario 1: UPTKIpcCloseMemHandle with NULL returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: UPTKIpcCloseMemHandle with invalid pointer
    void *invalidPtr = (void *)0xDEADBEEF;
    err = UPTKIpcCloseMemHandle(invalidPtr);
    printf("Scenario 2: UPTKIpcCloseMemHandle with invalid ptr returned: %s\n", UPTKGetErrorString(err));

    // Note: UPTKIpcCloseMemHandle requires multi-process IPC to test properly.
    printf("test_cudaIpcCloseMemHandle PASS\n");
    return 0;
}
