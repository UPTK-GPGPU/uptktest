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

    // Scene 1: Get capture info during capture
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    CHECK_CUDA(UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeRelaxed));

    void *d_a, *d_b;
    UPTKMalloc(&d_a, 1024);
    UPTKMalloc(&d_b, 1024);
    UPTKMemcpyAsync(d_b, d_a, 1024, UPTKMemcpyDefault, stream);

    UPTKStreamCaptureStatus status;
    unsigned long long id;
    CHECK_CUDA(UPTKStreamGetCaptureInfo(stream, &status, &id));

    UPTKGraph_t graph;
    CHECK_CUDA(UPTKStreamEndCapture(stream, &graph));

    // Scene 2: Check status when not capturing
    CHECK_CUDA(UPTKStreamGetCaptureInfo(0, &status, &id));

    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKStreamDestroy(stream);
    UPTKGraphDestroy(graph);

    printf("test_cudaStreamGetCaptureInfo PASS\n");
    return 0;
}
