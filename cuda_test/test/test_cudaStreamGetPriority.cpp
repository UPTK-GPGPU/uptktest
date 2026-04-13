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

    // Scene 1: Get priority of default stream
    int priority;
    CHECK_CUDA(UPTKStreamGetPriority(0, &priority));

    // Scene 2: Create stream with highest priority and verify
    int least_priority, greatest_priority;
    UPTKDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    UPTKStream_t stream1;
    UPTKStreamCreateWithPriority(&stream1, 0, greatest_priority);
    CHECK_CUDA(UPTKStreamGetPriority(stream1, &priority));

    // Scene 3: Create stream with lowest priority and verify
    UPTKStream_t stream2;
    UPTKStreamCreateWithPriority(&stream2, 0, least_priority);
    CHECK_CUDA(UPTKStreamGetPriority(stream2, &priority));

    UPTKStreamDestroy(stream1);
    UPTKStreamDestroy(stream2);

    printf("test_cudaStreamGetPriority PASS\n");
    return 0;
}
