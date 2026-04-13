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

    // Scene 1: Wait event on default stream with flags=0
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    CHECK_CUDA(UPTKStreamWaitEvent(0, event, 0));

    // Scene 2: Wait event on user stream with flags=0
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    CHECK_CUDA(UPTKStreamWaitEvent(stream, event, 0));

    // Scene 3: Record event and wait on it
    CHECK_CUDA(UPTKEventRecord(event, stream));
    CHECK_CUDA(UPTKStreamWaitEvent(stream, event, 0));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    UPTKEventDestroy(event);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamWaitEvent PASS\n");
    return 0;
}
