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

    // Scenario 1: Basic UPTKHostUnregister after registering memory
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
    printf("Scenario 1: UPTKHostUnregister after register PASS\n");

    // Scenario 2: UPTKHostUnregister on already unregistered memory (error case)
    h_ptr = malloc(size);
    if (h_ptr == NULL) {
        printf("malloc failed\n");
        return 1;
    }
    UPTKError_t err = UPTKHostUnregister(h_ptr);
    printf("Scenario 2: UPTKHostUnregister on unregistered memory: %s\n", UPTKGetErrorString(err));
    free(h_ptr);

    // Scenario 3: Error handling - NULL pointer
    err = UPTKHostUnregister(NULL);
    printf("Scenario 3: UPTKHostUnregister with NULL pointer: %s\n", UPTKGetErrorString(err));

    printf("test_cudaHostUnregister PASS\n");
    return 0;
}
