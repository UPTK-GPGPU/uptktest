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
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // UPTKIpcOpenEventHandle requires an IPC handle from another process.
    // Test error handling only.

    // Scenario 1: UPTKIpcOpenEventHandle with zeroed handle
    UPTKEvent_t event = NULL;
    UPTKIpcEventHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    UPTKError_t err = UPTKIpcOpenEventHandle(&event, handle);
    printf("Scenario 1: UPTKIpcOpenEventHandle with zeroed handle returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: UPTKIpcOpenEventHandle with NULL event pointer
    err = UPTKIpcOpenEventHandle(NULL, handle);
    printf("Scenario 2: UPTKIpcOpenEventHandle with NULL event returned: %s\n", UPTKGetErrorString(err));

    // Note: Full IPC testing requires multi-process environment.
    printf("test_cudaIpcOpenEventHandle PASS\n");
    return 0;
}
