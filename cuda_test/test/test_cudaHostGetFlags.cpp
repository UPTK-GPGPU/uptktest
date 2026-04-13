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

    // Scenario 1: Get flags from mapped host memory
    void *h_ptr = NULL;
    size_t size = 1024;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocMapped));

    unsigned int flags = 0;
    CHECK_CUDA(UPTKHostGetFlags(&flags, h_ptr));
    printf("Scenario 1: UPTKHostGetFlags returned flags: 0x%x\n", flags);
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scenario 2: Get flags from portable host memory
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocPortable));
    flags = 0;
    CHECK_CUDA(UPTKHostGetFlags(&flags, h_ptr));
    printf("Scenario 2: UPTKHostGetFlags for portable memory: 0x%x\n", flags);
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scenario 3: Error handling - NULL host pointer
    UPTKError_t err = UPTKHostGetFlags(&flags, NULL);
    printf("Scenario 3: NULL host pointer returned: %s\n", UPTKGetErrorString(err));

    // Scenario 4: Error handling - NULL flags pointer
    h_ptr = NULL;
    CHECK_CUDA(UPTKHostAlloc(&h_ptr, size, UPTKHostAllocDefault));
    err = UPTKHostGetFlags(NULL, h_ptr);
    printf("Scenario 4: NULL flags pointer returned: %s\n", UPTKGetErrorString(err));
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    printf("test_cudaHostGetFlags PASS\n");
    return 0;
}
