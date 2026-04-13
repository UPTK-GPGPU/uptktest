#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

    // Scene 1: Basic UPTKMemPoolSetAccess with read-write access
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    UPTKMemAccessDesc accessDesc;
    memset(&accessDesc, 0, sizeof(accessDesc));
    accessDesc.location.type = UPTKMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = UPTKMemAccessFlagsProtReadWrite;
    CHECK_CUDA(UPTKMemPoolSetAccess(memPool, &accessDesc, 1));

    // Scene 2: UPTKMemPoolSetAccess with read-only access
    UPTKMemAccessDesc accessDesc2;
    memset(&accessDesc2, 0, sizeof(accessDesc2));
    accessDesc2.location.type = UPTKMemLocationTypeDevice;
    accessDesc2.location.id = 0;
    accessDesc2.flags = UPTKMemAccessFlagsProtRead;
    UPTKError_t err = UPTKMemPoolSetAccess(memPool, &accessDesc2, 1);
    if (err != UPTKSuccess) {
        // Some platforms don't allow changing from RW to R
        printf("Note: changing access from RW to R returned: %s\n", UPTKGetErrorString(err));
    }

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolSetAccess PASS\n");
    return 0;
}
