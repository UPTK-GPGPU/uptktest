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

    UPTKError_t err = UPTKStreamQuery(0);

    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    void *d_ptr;
    UPTKMalloc(&d_ptr, 1024);
    UPTKMemsetAsync(d_ptr, 0, 1024, stream);
    err = UPTKStreamQuery(stream);

    UPTKStreamSynchronize(stream);
    CHECK_CUDA(UPTKStreamQuery(stream));

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamQuery_ptsz PASS\n");
    return 0;
}
