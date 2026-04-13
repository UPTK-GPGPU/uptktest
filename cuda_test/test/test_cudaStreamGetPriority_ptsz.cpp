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

    int priority;
    CHECK_CUDA(UPTKStreamGetPriority(0, &priority));

    int least_priority, greatest_priority;
    UPTKDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    UPTKStream_t stream;
    UPTKStreamCreateWithPriority(&stream, 0, greatest_priority);
    CHECK_CUDA(UPTKStreamGetPriority(stream, &priority));

    UPTKStreamDestroy(stream);

    printf("test_cudaStreamGetPriority_ptsz PASS\n");
    return 0;
}
