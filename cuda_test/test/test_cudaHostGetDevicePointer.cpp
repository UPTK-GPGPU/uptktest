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
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Get device pointer from mapped host memory
    void *h_ptr = NULL;
    size_t size = 1024;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocMapped));

    void *d_ptr = NULL;
    CHECK_CUDA(UPTKHostGetDevicePointer(&d_ptr, h_ptr, 0));
    if (d_ptr == NULL) {
        printf("UPTKHostGetDevicePointer returned NULL\n");
        return 1;
    }
    printf("Scenario 1: UPTKHostGetDevicePointer with flags=0, device ptr: %p\n", d_ptr);
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scenario 2: UPTKHostGetDevicePointer with cudaHostGetDevNoFlags (flags=0)
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocMapped));
    d_ptr = NULL;
    CHECK_CUDA(UPTKHostGetDevicePointer(&d_ptr, h_ptr, 0));
    printf("Scenario 2: UPTKHostGetDevicePointer with flags=0 PASS\n");
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scenario 3: Error handling - non-mapped pointer
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocDefault));
    d_ptr = NULL;
    UPTKError_t err = UPTKHostGetDevicePointer(&d_ptr, h_ptr, 0);
    printf("Scenario 3: Non-mapped pointer returned: %s\n", UPTKGetErrorString(err));
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scenario 4: Error handling - NULL host pointer
    err = UPTKHostGetDevicePointer(&d_ptr, NULL, 0);
    printf("Scenario 4: NULL host pointer returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaHostGetDevicePointer PASS\n");
    return 0;
}
