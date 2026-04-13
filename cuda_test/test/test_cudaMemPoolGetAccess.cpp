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

    // Scene 1: Basic UPTKMemPoolGetAccess
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    UPTKMemLocation loc;
    loc.type = UPTKMemLocationTypeDevice;
    loc.id = 0;
    enum UPTKMemAccessFlags flags;
    CHECK_CUDA(UPTKMemPoolGetAccess(&flags, memPool, &loc));

    // Scene 2: UPTKMemPoolGetAccess with set access first
    UPTKMemAccessDesc accessDesc;
    memset(&accessDesc, 0, sizeof(accessDesc));
    accessDesc.location.type = UPTKMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = UPTKMemAccessFlagsProtReadWrite;
    CHECK_CUDA(UPTKMemPoolSetAccess(memPool, &accessDesc, 1));
    enum UPTKMemAccessFlags flags2;
    CHECK_CUDA(UPTKMemPoolGetAccess(&flags2, memPool, &loc));

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolGetAccess PASS\n");
    return 0;
}
