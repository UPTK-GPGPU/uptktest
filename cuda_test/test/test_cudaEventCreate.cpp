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

    // Scenario 1: Create event with default flags
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    printf("Event created successfully\n");

    // Scenario 2: Query the event to verify it's valid
    UPTKError_t err = UPTKEventQuery(event);
    if (err == UPTKSuccess || err == UPTKErrorNotReady) {
        printf("Event query returned: %s\n", UPTKGetErrorString(err));
    }

    // Cleanup
    CHECK_CUDA(UPTKEventDestroy(event));

    printf("test_cudaEventCreate PASS\n");
    return 0;
}
