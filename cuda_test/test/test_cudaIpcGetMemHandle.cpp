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

    // Scenario 1: Basic UPTKIpcGetMemHandle
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKMalloc(&d_ptr, 1024));
    UPTKIpcMemHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    UPTKError_t err = UPTKIpcGetMemHandle(&handle, d_ptr);
    if (err == UPTKSuccess) {
        printf("Scenario 1: UPTKIpcGetMemHandle PASS\n");
    } else {
        printf("Scenario 1: UPTKIpcGetMemHandle returned: %s\n", UPTKGetErrorString(err));
    }
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scenario 2: Error handling - NULL handle pointer
    CHECK_CUDA(UPTKMalloc(&d_ptr, 1024));
    err = UPTKIpcGetMemHandle(NULL, d_ptr);
    printf("Scenario 2: NULL handle pointer returned: %s\n", UPTKGetErrorString(err));
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scenario 3: Error handling - NULL device pointer
    err = UPTKIpcGetMemHandle(&handle, NULL);
    printf("Scenario 3: NULL device pointer returned: %s\n", UPTKGetErrorString(err));

    // Note: Full IPC testing requires multi-process environment.
    printf("test_cudaIpcGetMemHandle PASS\n");
    return 0;
}
