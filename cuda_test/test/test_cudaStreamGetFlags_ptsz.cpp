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

    unsigned int flags;
    CHECK_CUDA(UPTKStreamGetFlags(0, &flags));

    UPTKStream_t stream;
    UPTKStreamCreateWithFlags(&stream, UPTKStreamNonBlocking);
    CHECK_CUDA(UPTKStreamGetFlags(stream, &flags));

    UPTKStreamDestroy(stream);

    printf("test_cudaStreamGetFlags_ptsz PASS\n");
    return 0;
}
