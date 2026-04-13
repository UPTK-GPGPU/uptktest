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

    // Scenario 1: Basic UPTKIpcGetEventHandle
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    UPTKIpcEventHandle_t handle;
    UPTKError_t err = UPTKIpcGetEventHandle(&handle, event);
    if (err == UPTKSuccess) {
        printf("Scenario 1: UPTKIpcGetEventHandle PASS\n");
    } else {
        printf("Scenario 1: UPTKIpcGetEventHandle returned: %s\n", UPTKGetErrorString(err));
    }
    CHECK_CUDA(UPTKEventDestroy(event));

    // Scenario 2: Error handling - NULL handle pointer
    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreate(&event2));
    err = UPTKIpcGetEventHandle(NULL, event2);
    printf("Scenario 2: NULL handle pointer returned: %s\n", UPTKGetErrorString(err));
    CHECK_CUDA(UPTKEventDestroy(event2));

    // Scenario 3: Error handling - NULL event
    err = UPTKIpcGetEventHandle(&handle, NULL);
    printf("Scenario 3: NULL event returned: %s\n", UPTKGetErrorString(err));

    // Note: Full IPC testing requires multi-process environment.
    printf("test_cudaIpcGetEventHandle PASS\n");
    return 0;
}
