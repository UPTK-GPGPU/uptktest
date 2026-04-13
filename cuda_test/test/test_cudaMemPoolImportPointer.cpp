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

    // Scene 1: Basic UPTKMemPoolImportPointer
    UPTKMemPool_t memPool;
    UPTKMemPoolProps poolProps;
    memset(&poolProps, 0, sizeof(poolProps));
    poolProps.allocType = UPTKMemAllocationTypePinned;
    poolProps.location.type = UPTKMemLocationTypeDevice;
    poolProps.location.id = 0;
    CHECK_CUDA(UPTKMemPoolCreate(&memPool, &poolProps));

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocFromPoolAsync(&d_ptr, 1024, memPool, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    UPTKMemPoolPtrExportData exportData;
    memset(&exportData, 0, sizeof(exportData));
    CHECK_CUDA(UPTKMemPoolExportPointer(&exportData, d_ptr));

    void *importedPtr = NULL;
    CHECK_CUDA(UPTKMemPoolImportPointer(&importedPtr, memPool, &exportData));

    CHECK_CUDA(UPTKFreeAsync(d_ptr, stream));
    CHECK_CUDA(UPTKStreamDestroy(stream));
    CHECK_CUDA(UPTKMemPoolDestroy(memPool));

    printf("test_cudaMemPoolImportPointer PASS\n");
    return 0;
}
