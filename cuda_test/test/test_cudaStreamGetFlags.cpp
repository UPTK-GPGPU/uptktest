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

    // Scene 1: Get flags from default stream
    unsigned int flags;
    CHECK_CUDA(UPTKStreamGetFlags(0, &flags));

    // Scene 2: Create stream with non-blocking flag and verify
    UPTKStream_t stream;
    UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking);
    CHECK_CUDA(UPTKStreamGetFlags(stream, &flags));

    // Scene 3: Create stream with default flag and verify
    UPTKStream_t stream2;
    UPTKStreamCreateWithFlags(&stream2, UPTKStreamDefault);
    CHECK_CUDA(UPTKStreamGetFlags(stream2, &flags));

    UPTKStreamDestroy(stream);
    UPTKStreamDestroy(stream2);

    printf("test_cudaStreamGetFlags PASS\n");
    return 0;
}
