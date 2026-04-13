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

    // Scene 1: Query event that has completed
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    CHECK_CUDA(UPTKEventRecord(event, 0));
    CHECK_CUDA(UPTKEventSynchronize(event));

    UPTKError_t status = UPTKEventQuery(event);
    if (status == UPTKSuccess) {
        printf("Event query: completed\n");
    } else {
        printf("Event query unexpected status: %s\n", UPTKGetErrorString(status));
        return 1;
    }

    CHECK_CUDA(UPTKEventDestroy(event));

    // Scene 2: Query event that may not be ready (no record called)
    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreate(&event2));

    status = UPTKEventQuery(event2);
    if (status == UPTKErrorNotReady) {
        printf("Event query: not ready (expected for unrecorded event)\n");
    } else if (status == UPTKSuccess) {
        printf("Event query: success (event already signaled)\n");
    } else {
        printf("Event query status: %s\n", UPTKGetErrorString(status));
    }

    CHECK_CUDA(UPTKEventDestroy(event2));

    printf("test_cudaEventQuery PASS\n");
    return 0;
}
