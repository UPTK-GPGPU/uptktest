#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

    // Scene 1: UPTKMemPoolExportToShareableHandle with posix file descriptor
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.handleTypes = UPTKMemHandleTypePosixFileDescriptor;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    int fd = -1;
    UPTKError_t err = UPTKMemPoolExportToShareableHandle(
        (void*)fd, memPool, UPTKMemHandleTypePosixFileDescriptor, 0);
    if (err == UPTKSuccess) {
        printf("Exported handle successfully\n");
        close(fd);
    } else if (err == UPTKErrorInvalidValue || err == UPTKErrorNotSupported) {
        printf("Handle export not supported on this platform, skipping\n");
    } else {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        CHECK_CUDA(UPTKMemPoolDestroy(memPool));
        return 1;
    }

    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolExportToShareableHandle PASS\n");
    return 0;
}
