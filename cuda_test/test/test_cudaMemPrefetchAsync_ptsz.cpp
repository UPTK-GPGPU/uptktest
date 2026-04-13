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

    // Scene 1: UPTKMemPrefetchAsync with per-thread default stream (ptsz variant)
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKMemPrefetchAsync(d_ptr, 1024 * sizeof(float), 0, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: cudaMemPrefetchAsync_ptsz with different size
    float *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr2, 2048 * sizeof(float)));
    CHECK_CUDA(UPTKMemPrefetchAsync(d_ptr2, 2048 * sizeof(float), 0, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());
    CHECK_CUDA(UPTKFree(d_ptr2));

    printf("test_cudaMemPrefetchAsync_ptsz PASS\n");
    return 0;
}
