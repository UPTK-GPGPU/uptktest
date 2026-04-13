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

    // Scene 1: UPTKMallocAsync with per-thread default stream (ptsz variant calls same function)
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocAsync(&d_ptr, 1024, UPTKStreamPerThread));
    CHECK_CUDA(UPTKFreeAsync(d_ptr, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());

    // Scene 2: cudaMallocAsync_ptsz with different size
    void *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocAsync(&d_ptr2, 2048, UPTKStreamPerThread));
    CHECK_CUDA(UPTKFreeAsync(d_ptr2, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());

    printf("test_cudaMallocAsync_ptsz PASS\n");
    return 0;
}
