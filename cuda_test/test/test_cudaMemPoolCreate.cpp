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

    // Scene 1: Basic UPTKMemPoolCreate
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    // Scene 2: UPTKMemPoolCreate with set access
    UPTKMemPool_t memPool2;
    UPTKMemPoolProps poolProps2;
    memset(&poolProps2, 0, sizeof(poolProps2));
    poolProps2.allocType = UPTKMemAllocationTypePinned;
    poolProps2.location.type = UPTKMemLocationTypeDevice;
    poolProps2.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool2, &poolProps2));

    UPTKMemAccessDesc accessDesc;
    memset(&accessDesc, 0, sizeof(accessDesc));
    accessDesc.location.type = UPTKMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = UPTKMemAccessFlagsProtReadWrite;
    CHECK_CUDA(UPTKMemPoolSetAccess(memPool2, &accessDesc, 1));

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));
    CHECK_CUDA(UPTKMemPoolDestroy(memPool2));

    printf("test_cudaMemPoolCreate PASS\n");
    return 0;
}
