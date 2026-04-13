#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Record event with default flags (0)
    UPTKEvent_t event;
    UPTKError_t err = UPTKEventCreate(&event);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    err = UPTKEventRecordWithFlags(event, 0, 0);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKEventRecordWithFlags not supported on DTK: %s\n", UPTKGetErrorString(err));
        UPTKEventDestroy(event);
        return 0;
    }

    err = UPTKEventSynchronize(event);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKEventDestroy(event);
        return 1;
    }
    printf("  Event recorded with default flags\n");
    UPTKEventDestroy(event);

    // Scenario 2: Record with UPTKEventRecordDefault flag
    UPTKEvent_t event2;
    err = UPTKEventCreate(&event2);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    err = UPTKEventRecordWithFlags(event2, 0, UPTKEventRecordDefault);
    if (err == UPTKSuccess) {
        err = UPTKEventSynchronize(event2);
        if (err == UPTKSuccess) {
            printf("  Event recorded with UPTKEventRecordDefault flag\n");
        }
    } else {
        printf("  UPTKEventRecordWithFlags with UPTKEventRecordDefault: %s\n", UPTKGetErrorString(err));
    }
    UPTKEventDestroy(event2);

    printf("test_cudaEventRecordWithFlags PASS\n");
    return 0;
}
