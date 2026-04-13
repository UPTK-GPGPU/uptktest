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

    // Scene 1: Basic async free on default stream
    float *d_a = NULL;
    CHECK_CUDA(UPTKMalloc(&d_a, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKFreeAsync(d_a, 0));
    printf("Basic async free on default stream succeeded\n");

    // Scene 2: Async free on explicit stream
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    float *d_b = NULL;
    CHECK_CUDA(UPTKMalloc(&d_b, 2048 * sizeof(float)));
    CHECK_CUDA(UPTKFreeAsync(d_b, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    printf("Async free on explicit stream succeeded\n");
    CHECK_CUDA(UPTKStreamDestroy(stream));

    printf("test_cudaFreeAsync PASS\n");
    return 0;
}
