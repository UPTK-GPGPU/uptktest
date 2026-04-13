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

    // Scenario 1: Create and destroy a single event
    UPTKEvent_t event1;
    CHECK_CUDA(UPTKEventCreate(&event1));
    CHECK_CUDA(UPTKEventDestroy(event1));
    printf("Single event created and destroyed\n");

    // Scenario 2: Create and destroy multiple events
    UPTKEvent_t events[3];
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(UPTKEventCreate(&events[i]));
    }
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(UPTKEventDestroy(events[i]));
    }
    printf("Multiple events created and destroyed\n");

    // Scenario 3: Create event with flags, record it, then destroy
    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreateWithFlags(&event2, UPTKEventDisableTiming));
    CHECK_CUDA(UPTKEventRecord(event2, 0));
    CHECK_CUDA(UPTKEventSynchronize(event2));
    CHECK_CUDA(UPTKEventDestroy(event2));
    printf("Event recorded, synchronized, and destroyed\n");

    printf("test_cudaEventDestroy PASS\n");
    return 0;
}
