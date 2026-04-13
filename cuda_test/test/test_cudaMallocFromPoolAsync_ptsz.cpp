#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>

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

    // Scene 1: UPTKMallocFromPoolAsync with per-thread default stream (ptsz variant)
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    void *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocFromPoolAsync(&d_ptr, 1024, memPool, UPTKStreamPerThread));
    CHECK_CUDA(UPTKFreeAsync(d_ptr, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());

    // Scene 2: cudaMallocFromPoolAsync_ptsz with different size
    void *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocFromPoolAsync(&d_ptr2, 2048, memPool, UPTKStreamPerThread));
    CHECK_CUDA(UPTKFreeAsync(d_ptr2, UPTKStreamPerThread));
    CHECK_CUDA(UPTKDeviceSynchronize());

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMallocFromPoolAsync_ptsz PASS\n");
    return 0;
}
