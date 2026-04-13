#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdlib.h>
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

    // Scenario 1: Basic UPTKHostRegister with default flags
    size_t size = 4096;
    void *h_ptr = malloc(size);
    if (h_ptr == NULL) {
        printf("malloc failed\n");
        return 1;
    }
    memset(h_ptr, 0, size);
    CHECK_CUDA(UPTKHostRegister(h_ptr, size, UPTKHostRegisterDefault));
    CHECK_CUDA(UPTKHostUnregister(h_ptr));
    free(h_ptr);
    printf("Scenario 1: UPTKHostRegister with default flags PASS\n");

    // Scenario 2: UPTKHostRegister with UPTKHostRegisterPortable
    h_ptr = malloc(size);
    if (h_ptr == NULL) {
        printf("malloc failed\n");
        return 1;
    }
    memset(h_ptr, 0, size);
    CHECK_CUDA(UPTKHostRegister(h_ptr, size, UPTKHostRegisterPortable));
    CHECK_CUDA(UPTKHostUnregister(h_ptr));
    free(h_ptr);
    printf("Scenario 2: UPTKHostRegister with UPTKHostRegisterPortable PASS\n");

    // Scenario 3: UPTKHostRegister with UPTKHostRegisterMapped
    h_ptr = malloc(size);
    if (h_ptr == NULL) {
        printf("malloc failed\n");
        return 1;
    }
    memset(h_ptr, 0, size);
    CHECK_CUDA(UPTKHostRegister(h_ptr, size, UPTKHostRegisterMapped));
    void *d_ptr = NULL;
    CHECK_CUDA(UPTKHostGetDevicePointer(&d_ptr, h_ptr, 0));
    if (d_ptr == NULL) {
        printf("UPTKHostGetDevicePointer returned NULL\n");
        return 1;
    }
    CHECK_CUDA(UPTKHostUnregister(h_ptr));
    free(h_ptr);
    printf("Scenario 3: UPTKHostRegister with UPTKHostRegisterMapped PASS\n");

    // Scenario 4: Error handling - NULL pointer
    UPTKError_t err = UPTKHostRegister(NULL, size, UPTKHostRegisterDefault);
    printf("Scenario 4: NULL pointer returned: %s\n", UPTKGetErrorString(err));

    printf("test_cudaHostRegister PASS\n");
    return 0;
}
