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

    // Scene 1: Basic UPTKMemPoolSetAttribute with UPTKMemPoolReuseFollowEventDependencies
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    int enable = 1;
    CHECK_CUDA(UPTKMemPoolSetAttribute(memPool, UPTKMemPoolReuseFollowEventDependencies, &enable));

    // Scene 2: UPTKMemPoolSetAttribute with UPTKMemPoolReuseAllowOpportunistic
    int allow = 0;
    CHECK_CUDA(UPTKMemPoolSetAttribute(memPool, UPTKMemPoolReuseAllowOpportunistic, &allow));

    // Scene 3: UPTKMemPoolSetAttribute with UPTKMemPoolAttrReleaseThreshold
    uint64_t threshold = 1024 * 1024;
    CHECK_CUDA(UPTKMemPoolSetAttribute(memPool, UPTKMemPoolAttrReleaseThreshold, &threshold));

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolSetAttribute PASS\n");
    return 0;
}
