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

    // Scene 1: Create stream with default flags (0)
    UPTKStream_t stream1;
    CHECK_CUDA(UPTKStreamCreateWithFlags(&stream1, 0));
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream1);
    UPTKStreamSynchronize(stream1);

    // Scene 2: Create stream with UPTKStreamNonBlocking flag
    UPTKStream_t stream2;
    CHECK_CUDA(UPTKStreamCreateWithFlags(&stream2, UPTKStreamNonBlocking));
    UPTKMemsetAsync(d_ptr, 0xFF, 1024, stream2);
    UPTKStreamSynchronize(stream2);

    // Scene 3: Create stream with UPTKStreamDefault flag
    UPTKStream_t stream3;
    CHECK_CUDA(UPTKStreamCreateWithFlags(&stream3, UPTKStreamDefault));
    UPTKMemsetAsync(d_ptr, 0, 1024, stream3);
    UPTKStreamSynchronize(stream3);

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream1);
    UPTKStreamDestroy(stream2);
    UPTKStreamDestroy(stream3);

    printf("test_cudaStreamCreateWithFlags PASS\n");
    return 0;
}
