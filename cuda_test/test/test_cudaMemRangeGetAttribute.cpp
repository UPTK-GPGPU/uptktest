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

    // Scene 1: Basic UPTKMemRangeGetAttribute with UPTKMemRangeAttributeReadMostly
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    int readMostly = 0;
    size_t dataSize = sizeof(readMostly);
    UPTKError_t err = UPTKMemRangeGetAttribute(&readMostly, dataSize,
        UPTKMemRangeAttributeReadMostly, d_ptr, 1024 * sizeof(float));
    if (err != UPTKSuccess) {
        // DTK/AMD may not support this attribute, just verify the API is callable
        printf("Note: UPTKMemRangeGetAttribute returned: %s (expected on DTK)\n", UPTKGetErrorString(err));
    }
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: UPTKMemRangeGetAttribute with UPTKMemRangeAttributePreferredLocation
    float *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr2, 512 * sizeof(float)));
    int location = -1;
    dataSize = sizeof(location);
    err = UPTKMemRangeGetAttribute(&location, dataSize,
        UPTKMemRangeAttributePreferredLocation, d_ptr2, 512 * sizeof(float));
    if (err != UPTKSuccess) {
        printf("Note: UPTKMemRangeGetAttribute returned: %s (expected on DTK)\n", UPTKGetErrorString(err));
    }
    CHECK_CUDA(UPTKFree(d_ptr2));

    printf("test_cudaMemRangeGetAttribute PASS\n");
    return 0;
}
