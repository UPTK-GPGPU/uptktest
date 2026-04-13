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

    CHECK_CUDA(UPTKStreamSynchronize(0));

    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream);
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamSynchronize_ptsz PASS\n");
    return 0;
}
