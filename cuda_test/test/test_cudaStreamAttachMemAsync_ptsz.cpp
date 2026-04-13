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
    void *d_ptr = NULL;
    UPTKMallocManaged(&d_ptr, 1024);
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 1024, UPTKMemAttachSingle));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 1024, UPTKMemAttachHost));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamAttachMemAsync_ptsz PASS\n");
    return 0;
}
