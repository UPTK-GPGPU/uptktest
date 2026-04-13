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

    // Scene 1: Get attributes for device pointer
    int *d_data = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data, 256 * sizeof(int)));

    UPTKPointerAttributes attr;
    memset(&attr, 0, sizeof(attr));
    CHECK_CUDA(UPTKPointerGetAttributes(&attr, d_data));

    int pass = 1;
    // Verify the pointer is a device pointer
    if (attr.type != UPTKMemoryTypeDevice) {
        printf("Scene 1 FAIL: expected UPTKMemoryTypeDevice, got %d\n", attr.type);
        pass = 0;
    }

    // Scene 2: Get attributes for pinned host pointer
    int *h_pinned = NULL;
    CHECK_CUDA(UPTKMallocHost((void **)&h_pinned, 256 * sizeof(int)));
    memset(&attr, 0, sizeof(attr));
    CHECK_CUDA(UPTKPointerGetAttributes(&attr, h_pinned));
    // Pinned host pointer should have type UPTKMemoryTypeHost
    if (attr.type != UPTKMemoryTypeHost) {
        printf("Scene 2 FAIL: expected UPTKMemoryTypeHost, got %d\n", attr.type);
        pass = 0;
    }
    UPTKFreeHost(h_pinned);

    // Scene 3: Get attributes for managed memory
    int *d_managed = NULL;
    CHECK_CUDA(UPTKMallocManaged(&d_managed, 256 * sizeof(int)));
    memset(&attr, 0, sizeof(attr));
    CHECK_CUDA(UPTKPointerGetAttributes(&attr, d_managed));
    if (attr.type != UPTKMemoryTypeManaged && attr.type != UPTKMemoryTypeDevice) {
        printf("Scene 3 FAIL: expected UPTKMemoryTypeManaged or UPTKMemoryTypeDevice, got %d\n", attr.type);
        pass = 0;
    }
    UPTKFree(d_managed);

    UPTKFree(d_data);

    if (pass) {
        printf("test_cudaPointerGetAttributes PASS\n");
        return 0;
    } else {
        printf("test_cudaPointerGetAttributes FAIL: memory type mismatch\n");
        return 1;
    }
}
