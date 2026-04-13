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

    // Scene 1: Synchronize default stream
    CHECK_CUDA(UPTKStreamSynchronize(0));

    // Scene 2: Create stream, add work, synchronize
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream);
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Synchronize non-blocking stream
    UPTKStream_t stream2;
    UPTKStreamCreateWithFlags(&stream2, UPTKStreamNonBlocking);
    UPTKMemsetAsync(d_ptr, 0xFF, 1024, stream2);
    CHECK_CUDA(UPTKStreamSynchronize(stream2));

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);
    UPTKStreamDestroy(stream2);

    printf("test_cudaStreamSynchronize PASS\n");
    return 0;
}
