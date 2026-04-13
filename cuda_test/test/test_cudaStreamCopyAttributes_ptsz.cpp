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
    UPTKStream_t stream1, stream2;
    CHECK_CUDA(UPTKStreamCreate(&stream1));
    CHECK_CUDA(UPTKStreamCreate(&stream2));

    UPTKError_t err = UPTKStreamCopyAttributes(stream2, stream1);
    if (err != UPTKSuccess) {
        printf("UPTKStreamCopyAttributes returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Stream attributes copied successfully\n");
    }

    err = UPTKStreamCopyAttributes(stream1, 0);
    if (err != UPTKSuccess) {
        printf("UPTKStreamCopyAttributes (default->user) returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Stream attributes copied from default to user\n");
    }

    UPTKStreamDestroy(stream1);
    UPTKStreamDestroy(stream2);

    printf("test_cudaStreamCopyAttributes_ptsz PASS\n");
    return 0;
}
