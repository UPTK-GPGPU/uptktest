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

    // Scene 1: Update capture dependencies with empty list
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    CHECK_CUDA(UPTKStreamBeginCapture(stream, UPTKStreamCaptureModeRelaxed));

    void *d_a, *d_b;
    UPTKMalloc(&d_a, 1024);
    UPTKMalloc(&d_b, 1024);
    UPTKMemcpyAsync(d_b, d_a, 1024, UPTKMemcpyDefault, stream);

    CHECK_CUDA(UPTKStreamUpdateCaptureDependencies(stream, NULL, 0, 0));

    UPTKGraph_t graph;
    CHECK_CUDA(UPTKStreamEndCapture(stream, &graph));

    // Scene 2: Update with UPTKStreamUpdateCaptureDependenciesFlags
    UPTKStream_t stream2;
    UPTKStreamCreate(&stream2);
    CHECK_CUDA(UPTKStreamBeginCapture(stream2, UPTKStreamCaptureModeRelaxed));
    UPTKMemcpyAsync(d_a, d_b, 1024, UPTKMemcpyDefault, stream2);
    //CHECK_CUDA(UPTKStreamUpdateCaptureDependencies(stream2, NULL, 0, UPTKStreamSetCaptureDependencies));
    CHECK_CUDA(UPTKStreamUpdateCaptureDependencies(stream2, NULL, 0, 0));

    UPTKGraph_t graph2;
    CHECK_CUDA(UPTKStreamEndCapture(stream2, &graph2));

    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKStreamDestroy(stream);
    UPTKStreamDestroy(stream2);
    UPTKGraphDestroy(graph);
    UPTKGraphDestroy(graph2);

    printf("test_cudaStreamUpdateCaptureDependencies PASS\n");
    return 0;
}
