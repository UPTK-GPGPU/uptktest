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

    // Scene 1: Basic attach managed memory to stream
    void *d_ptr = NULL;
    UPTKMallocManaged(&d_ptr, 1024);
    UPTKStream_t stream;
    UPTKStreamCreate(&stream);
    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 1024, UPTKMemAttachSingle));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 2: Attach with UPTKMemAttachHost flag
    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 1024, UPTKMemAttachHost));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Attach with length 0 (should succeed, means attach all remaining)
    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 0, UPTKMemAttachSingle));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 4: Attach with global flag
    CHECK_CUDA(UPTKStreamAttachMemAsync(stream, d_ptr, 1024, UPTKMemAttachGlobal));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    UPTKFree(d_ptr);
    UPTKStreamDestroy(stream);

    printf("test_cudaStreamAttachMemAsync PASS\n");
    return 0;
}
