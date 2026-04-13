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

    // Scene 1: Basic UPTKMallocHost
    void *h_ptr = NULL;
    CHECK_CUDA(UPTKMallocHost(&h_ptr, 1024));
    for (int i = 0; i < 256; i++) ((float*)h_ptr)[i] = (float)i;
    CHECK_CUDA(UPTKFreeHost(h_ptr));

    // Scene 2: UPTKMallocHost with larger allocation
    void *h_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocHost(&h_ptr2, 4096 * 1024));
    for (int i = 0; i < 1024; i++) ((double*)h_ptr2)[i] = (double)i;
    CHECK_CUDA(UPTKFreeHost(h_ptr2));

    // Scene 3: UPTKMallocHost with size 0 (boundary condition)
    void *h_ptr3 = NULL;
    UPTKError_t err = UPTKMallocHost(&h_ptr3, 0);
    if (err == UPTKSuccess || err == UPTKErrorInvalidValue) {
        if (err == UPTKSuccess) {
            CHECK_CUDA(UPTKFreeHost(h_ptr3));
        }
    } else {
        printf("CUDA error: unexpected error for size 0\n");
        return 1;
    }

    printf("test_cudaMallocHost PASS\n");
    return 0;
}
