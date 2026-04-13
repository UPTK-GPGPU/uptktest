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

    // Scene 1: Basic event record on default stream using _ptsz variant
    UPTKEvent_t event;
    CHECK_CUDA(UPTKEventCreate(&event));
    CHECK_CUDA(UPTKEventRecord(event, 0));
    CHECK_CUDA(UPTKEventSynchronize(event));
    printf("Event recorded on default stream (_ptsz variant)\n");
    CHECK_CUDA(UPTKEventDestroy(event));

    // Scene 2: Event record on explicit stream using _ptsz variant
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    UPTKEvent_t event2;
    CHECK_CUDA(UPTKEventCreate(&event2));
    CHECK_CUDA(UPTKEventRecord(event2, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    printf("Event recorded on explicit stream (_ptsz variant)\n");
    CHECK_CUDA(UPTKEventDestroy(event2));
    CHECK_CUDA(UPTKStreamDestroy(stream));

    printf("test_cudaEventRecord_ptsz PASS\n");
    return 0;
}
