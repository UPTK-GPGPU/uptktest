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

    // Scene 1: Create stream with default priority
    UPTKStream_t stream1;
    CHECK_CUDA(UPTKStreamCreateWithPriority(&stream1, 0, 0));
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream1);
    UPTKStreamSynchronize(stream1);

    // Scene 2: Create stream with highest priority
    int least_priority, greatest_priority;
    UPTKDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    UPTKStream_t stream2;
    CHECK_CUDA(UPTKStreamCreateWithPriority(&stream2, 0, greatest_priority));
    UPTKMemsetAsync(d_ptr, 0xFF, 1024, stream2);
    UPTKStreamSynchronize(stream2);

    // Scene 3: Create stream with lowest priority
    UPTKStream_t stream3;
    CHECK_CUDA(UPTKStreamCreateWithPriority(&stream3, 0, least_priority));
    UPTKMemsetAsync(d_ptr, 0, 1024, stream3);
    UPTKStreamSynchronize(stream3);

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream1);
    UPTKStreamDestroy(stream2);
    UPTKStreamDestroy(stream3);

    printf("test_cudaStreamCreateWithPriority PASS\n");
    return 0;
}
