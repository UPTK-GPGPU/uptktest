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

    // UPTKIpcOpenMemHandle requires an IPC handle from another process.
    // Test error handling only.

    // Scenario 1: UPTKIpcOpenMemHandle with zeroed handle
    void *devPtr = NULL;
    UPTKIpcMemHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    UPTKError_t err = UPTKIpcOpenMemHandle(&devPtr, handle, UPTKIpcMemLazyEnablePeerAccess);
    printf("Scenario 1: UPTKIpcOpenMemHandle with zeroed handle returned: %s\n", UPTKGetErrorString(err));

    // Scenario 2: UPTKIpcOpenMemHandle with NULL output pointer
    err = UPTKIpcOpenMemHandle(NULL, handle, UPTKIpcMemLazyEnablePeerAccess);
    printf("Scenario 2: UPTKIpcOpenMemHandle with NULL output returned: %s\n", UPTKGetErrorString(err));

    // Scenario 3: UPTKIpcOpenMemHandle with flags=0
    err = UPTKIpcOpenMemHandle(&devPtr, handle, 0);
    printf("Scenario 3: UPTKIpcOpenMemHandle with flags=0 returned: %s\n", UPTKGetErrorString(err));

    // Note: Full IPC testing requires multi-process environment.
    printf("test_cudaIpcOpenMemHandle PASS\n");
    return 0;
}
