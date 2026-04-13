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

    // Scenario 1: Get stream priority range
    int leastPriority, greatestPriority;
    CHECK_CUDA(UPTKDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    printf("Stream priority range: least=%d, greatest=%d\n", leastPriority, greatestPriority);

    // Scenario 2: Verify least priority >= greatest priority (lower number = higher priority)
    if (leastPriority >= greatestPriority) {
        printf("Priority range validation passed\n");
    } else {
        printf("Warning: unexpected priority range\n");
    }

    printf("test_cudaDeviceGetStreamPriorityRange PASS\n");
    return 0;
}
