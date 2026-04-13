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

    // Scene 1: Basic UPTKMemPrefetchAsync with explicit stream
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    CHECK_CUDA(UPTKMemPrefetchAsync(d_ptr, 1024 * sizeof(float), 0, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    CHECK_CUDA(UPTKFree(d_ptr));
    CHECK_CUDA(UPTKStreamDestroy(stream));

    // Scene 2: UPTKMemPrefetchAsync with default stream (0)
    float *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr2, 512 * sizeof(float)));
    CHECK_CUDA(UPTKMemPrefetchAsync(d_ptr2, 512 * sizeof(float), 0, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFree(d_ptr2));

    printf("test_cudaMemPrefetchAsync PASS\n");
    return 0;
}
