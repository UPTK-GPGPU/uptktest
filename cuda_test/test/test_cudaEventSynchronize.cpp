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

    // Scene 1: Synchronize a recorded event
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    CHECK_CUDA(UPTKEventRecord(event, 0));
    CHECK_CUDA(UPTKEventSynchronize(event));
    printf("Event synchronized after record\n");
    CHECK_CUDA(UPTKEventDestroy(event));

    // Scene 2: Synchronize event with GPU work in between
    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreate(&event2));

    float *d_a = NULL;
    CHECK_CUDA(UPTKMalloc(&d_a, 4096 * sizeof(float)));

    CHECK_CUDA(UPTKEventRecord(event2, 0));
    CHECK_CUDA(UPTKMemset(d_a, 0, 4096 * sizeof(float)));
    CHECK_CUDA(UPTKEventSynchronize(event2));
    printf("Event synchronized with GPU work\n");

    CHECK_CUDA(UPTKFree(d_a));
    CHECK_CUDA(UPTKEventDestroy(event2));

    printf("test_cudaEventSynchronize PASS\n");
    return 0;
}
