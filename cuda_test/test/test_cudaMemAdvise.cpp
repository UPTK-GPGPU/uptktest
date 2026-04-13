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

    // Scene 1: Basic UPTKMemAdvise with UPTKMemAdviseSetReadMostly
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKMemAdvise(d_ptr, 1024 * sizeof(float), UPTKMemAdviseSetReadMostly, 0));
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: UPTKMemAdvise with UPTKMemAdviseUnsetReadMostly
    float *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr2, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKMemAdvise(d_ptr2, 1024 * sizeof(float), UPTKMemAdviseSetReadMostly, 0));
    CHECK_CUDA(UPTKMemAdvise(d_ptr2, 1024 * sizeof(float), UPTKMemAdviseUnsetReadMostly, 0));
    CHECK_CUDA(UPTKFree(d_ptr2));

    // Scene 3: UPTKMemAdvise with UPTKMemAdviseSetPreferredLocation
    float *d_ptr3 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr3, 512 * sizeof(float)));
    CHECK_CUDA(UPTKMemAdvise(d_ptr3, 512 * sizeof(float), UPTKMemAdviseSetPreferredLocation, 0));
    CHECK_CUDA(UPTKFree(d_ptr3));

    printf("test_cudaMemAdvise PASS\n");
    return 0;
}
