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

    // Scene 1: Create and destroy a stream
    UPTKStream_t stream1;
    UPTKStreamCreate(&stream1);
    CHECK_CUDA(UPTKStreamDestroy(stream1));

    // Scene 2: Create with flags and destroy
    UPTKStream_t stream2;
    UPTKStreamCreateWithFlags(&stream2, UPTKStreamNonBlocking);
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream2);
    UPTKStreamSynchronize(stream2);
    CHECK_CUDA(UPTKStreamDestroy(stream2));

    UPTKFree(d_ptr);

    printf("test_cudaStreamDestroy PASS\n");
    return 0;
}
