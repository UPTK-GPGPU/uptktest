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
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Basic UPTKHostAlloc with default flags
    void *h_ptr = NULL;
    size_t size = 1024;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocDefault));
    if (h_ptr == NULL) {
        printf("UPTKHostAlloc returned NULL pointer\n");
        return 1;
    }
    memset(h_ptr, 0xAA, size);
    CHECK_CUDA(UPTKFreeHost(h_ptr));
    printf("Scenario 1: UPTKHostAlloc with default flags PASS\n");

    // Scenario 2: UPTKHostAlloc with UPTKHostAllocPortable
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocPortable));
    if (h_ptr == NULL) {
        printf("UPTKHostAlloc returned NULL pointer\n");
        return 1;
    }
    memset(h_ptr, 0xBB, size);
    CHECK_CUDA(UPTKFreeHost(h_ptr));
    printf("Scenario 2: UPTKHostAlloc with UPTKHostAllocPortable PASS\n");

    // Scenario 3: UPTKHostAlloc with UPTKHostAllocMapped
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocMapped));
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKHostGetDevicePointer(&d_ptr, h_ptr, 0));
    if (d_ptr == NULL) {
        printf("UPTKHostGetDevicePointer returned NULL\n");
        return 1;
    }
    CHECK_CUDA(UPTKFreeHost(h_ptr));
    printf("Scenario 3: UPTKHostAlloc with UPTKHostAllocMapped PASS\n");

    // Scenario 4: UPTKHostAlloc with UPTKHostAllocWriteCombined
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocWriteCombined));
    memset(h_ptr, 0xCC, size);
    CHECK_CUDA(UPTKFreeHost(h_ptr));
    printf("Scenario 4: UPTKHostAlloc with UPTKHostAllocWriteCombined PASS\n");

    // Scenario 5: Error handling - zero size allocation
    h_ptr = NULL;
    UPTKError_t err = UPTKHostAlloc(&h_ptr, 0, UPTKHostAllocDefault);
    printf("Scenario 5: UPTKHostAlloc with size=0 returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaHostAlloc PASS\n");
    return 0;
}
