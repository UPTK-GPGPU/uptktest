#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <cstdint>

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

    // Scene 1: Basic UPTKMemPoolGetAttribute with cudaMemPoolAttrReuseFollowEventDependencies
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    int value = 0;
    CHECK_CUDA(UPTKMemPoolGetAttribute(memPool, UPTKMemPoolReuseFollowEventDependencies, &value));

    // Scene 2: UPTKMemPoolGetAttribute with UPTKMemPoolAttrReservedMemCurrent
    uint64_t reserved = 0;
    CHECK_CUDA(UPTKMemPoolGetAttribute(memPool, UPTKMemPoolAttrReservedMemCurrent, &reserved));

    // Scene 3: UPTKMemPoolGetAttribute with UPTKMemPoolAttrUsedMemCurrent
    uint64_t used = 0;
    CHECK_CUDA(UPTKMemPoolGetAttribute(memPool, UPTKMemPoolAttrUsedMemCurrent, &used));

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolGetAttribute PASS\n");
    return 0;
}
