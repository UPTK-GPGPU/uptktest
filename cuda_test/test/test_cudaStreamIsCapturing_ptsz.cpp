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

    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    UPTKStreamCaptureStatus status;
    CHECK_CUDA(UPTKStreamIsCapturing(stream, &status));

    CHECK_CUDA(UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeRelaxed));
    void *d_a, *d_b;
    UPTKMalloc(&d_a, 1024);
    UPTKMalloc(&d_b, 1024);
    UPTKMemcpyAsync(d_b, d_a, 1024, UPTKMemcpyDefault, stream);
    CHECK_CUDA(UPTKStreamIsCapturing(stream, &status));

    UPTKGraph_t graph;
    CHECK_CUDA(UPTKStreamEndCapture(stream, &graph));

    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKStreamDestroy(stream);
    UPTKGraphDestroy(graph);

    printf("test_cudaStreamIsCapturing_ptsz PASS\n");
    return 0;
}
