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

    // Scene 1: Basic UPTKMallocPitch
    float *d_ptr = NULL;
    size_t pitch = 0;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr, &pitch, 256 * sizeof(float), 128));
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: UPTKMallocPitch with different dimensions
    int *d_ptr2 = NULL;
    size_t pitch2 = 0;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr2, &pitch2, 64 * sizeof(int), 64));
    CHECK_CUDA(UPTKFree(d_ptr2));

    // Scene 3: UPTKMallocPitch with small dimensions
    char *d_ptr3 = NULL;
    size_t pitch3 = 0;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_ptr3, &pitch3, 16, 8));
    CHECK_CUDA(UPTKFree(d_ptr3));

    printf("test_cudaMallocPitch PASS\n");
    return 0;
}
