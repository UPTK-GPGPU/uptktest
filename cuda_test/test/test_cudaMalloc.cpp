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

    // Scenario 1: Basic UPTKMalloc
    void *d_ptr = NULL;
    size_t size = 1024;
    CHECK_CUDA(UPTKMalloc(&d_ptr, size));
    if (d_ptr == NULL) {
        printf("UPTKMalloc returned NULL pointer\n");
        return 1;
    }
    CHECK_CUDA(UPTKFree(d_ptr));
    printf("Scenario 1: UPTKMalloc basic PASS\n");

    // Scenario 2: UPTKMalloc with small size
    d_ptr = NULL;
    CHECK_CUDA(UPTKMalloc(&d_ptr, 1));
    CHECK_CUDA(UPTKFree(d_ptr));
    printf("Scenario 2: UPTKMalloc with size=1 PASS\n");

    // Scenario 3: UPTKMalloc with larger size (1MB)
    d_ptr = NULL;
    CHECK_CUDA(UPTKMalloc(&d_ptr, 1024 * 1024));
    CHECK_CUDA(UPTKFree(d_ptr));
    printf("Scenario 3: UPTKMalloc with 1MB PASS\n");

    // Scenario 4: Error handling - NULL pointer
    UPTKError_t err = UPTKMalloc(NULL, size);
    printf("Scenario 4: NULL pointer returned: %s\n", UPTKGetErrorString(err));

    // Scenario 5: UPTKMalloc with size 0 (implementation-defined behavior)
    d_ptr = NULL;
    err = UPTKMalloc(&d_ptr, 0);
    printf("Scenario 5: UPTKMalloc with size=0 returned: %s, ptr=%p\n", UPTKGetErrorString(err), d_ptr);
    if (d_ptr != NULL) {
        CHECK_CUDA(UPTKFree(d_ptr));
    }

    printf("test_cudaMalloc PASS\n");
    return 0;
}
