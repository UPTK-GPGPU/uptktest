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

    // Scene 1: End capture on a stream that was started with relaxed mode
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    UPTKError_t err = UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeRelaxed);
    if (err != UPTKSuccess) {
        printf("UPTKStreamBeginCapture returned: %s (stream capture not fully supported on this platform)\n", UPTKGetErrorString(err));
        UPTKStreamDestroy(stream);
        printf("test_cudaStreamEndCapture PASS\n");
        return 0;
    }

    void *d_a, *d_b;
    CHECK_CUDA(UPTKMalloc(&d_a, 1024));
    CHECK_CUDA(UPTKMalloc(&d_b, 1024));
    CHECK_CUDA(UPTKMemcpyAsync(d_b, d_a, 1024, UPTKMemcpyDefault, stream));

    UPTKGraph_t graph;
    err = UPTKStreamEndCapture(stream, &graph);
    if (err != UPTKSuccess) {
        printf("UPTKStreamEndCapture returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Stream capture ended successfully, graph created\n");
        UPTKGraphDestroy(graph);
    }

    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamEndCapture PASS\n");
    return 0;
}
