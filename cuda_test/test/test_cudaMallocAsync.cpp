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

    // Scene 1: Basic UPTKMallocAsync with stream
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocAsync(&d_ptr, 1024, stream));
    CHECK_CUDA(UPTKFreeAsync(d_ptr, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));
    CHECK_CUDA(UPTKStreamDestroy(stream));

    // Scene 2: UPTKMallocAsync with default stream
    void *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocAsync(&d_ptr2, 2048, 0));
    CHECK_CUDA(UPTKFreeAsync(d_ptr2, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());

    // Scene 3: UPTKMallocAsync with larger allocation
    UPTKStream_t stream2;
    CHECK_CUDA(UPTKStreamCreate(&stream2));
    void *d_ptr3 = NULL;
    CHECK_CUDA(UPTKMallocAsync(&d_ptr3, 4096 * 4096, stream2));
    CHECK_CUDA(UPTKFreeAsync(d_ptr3, stream2));
    CHECK_CUDA(UPTKStreamSynchronize(stream2));
    CHECK_CUDA(UPTKStreamDestroy(stream2));

    printf("test_cudaMallocAsync PASS\n");
    return 0;
}
