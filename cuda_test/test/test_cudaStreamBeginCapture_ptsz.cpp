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

    // _ptsz variant calls the same function as the base version
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    CHECK_CUDA(UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeRelaxed));

    void *d_a, *d_b;
    UPTKMalloc(&d_a, 1024);
    UPTKMalloc(&d_b, 1024);
    UPTKMemcpyAsync(d_b, d_a, 1024, UPTKMemcpyDefault, stream);

    UPTKGraph_t graph;
    CHECK_CUDA(UPTKStreamEndCapture(stream, &graph));

    void *d_c;
    UPTKMalloc(&d_c, 512);
    CHECK_CUDA(UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeThreadLocal));
    UPTKMemcpyAsync(d_c, d_a, 512, UPTKMemcpyDefault, stream);
    UPTKGraph_t graph2;
    CHECK_CUDA(UPTKStreamEndCapture(stream, &graph2));

    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    UPTKStreamDestroy(stream);
    UPTKGraphDestroy(graph);
    UPTKGraphDestroy(graph2);

    printf("test_cudaStreamBeginCapture_ptsz PASS\n");
    return 0;
}
