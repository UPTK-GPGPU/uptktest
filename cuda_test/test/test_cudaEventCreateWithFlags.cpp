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

    // Scenario 1: Create event with UPTKEventDefault flag (0)
    UPTKEvent_t event1;
    CHECK_CUDA(UPTKEventCreateWithFlags(&event1, UPTKEventDefault));
    printf("Event with UPTKEventDefault created\n");
    CHECK_CUDA(UPTKEventDestroy(event1));

    // Scenario 2: Create event with UPTKEventBlockingSync flag
    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreateWithFlags(&event2, UPTKEventBlockingSync));
    printf("Event with UPTKEventBlockingSync created\n");
    CHECK_CUDA(UPTKEventDestroy(event2));

    // Scenario 3: Create event with UPTKEventDisableTiming flag
    UPTKEvent_t event3;
    CHECK_CUDA(UPTKEventCreateWithFlags(&event3, UPTKEventDisableTiming));
    printf("Event with UPTKEventDisableTiming created\n");
    CHECK_CUDA(UPTKEventDestroy(event3));

    // Scenario 4: Create event with combined flags
    UPTKEvent_t event4;
    CHECK_CUDA(UPTKEventCreateWithFlags(&event4, UPTKEventBlockingSync | UPTKEventDisableTiming));
    printf("Event with combined flags created\n");
    CHECK_CUDA(UPTKEventDestroy(event4));

    printf("test_cudaEventCreateWithFlags PASS\n");
    return 0;
}
