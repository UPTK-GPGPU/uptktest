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

    // Scene 1: Basic UPTKMemGetInfo
    size_t free_mem = 0, total_mem = 0;
    CHECK_CUDA(UPTKMemGetInfo(&free_mem, &total_mem));
    printf("Free: %zu, Total: %zu\n", free_mem, total_mem);

    // Scene 2: UPTKMemGetInfo after allocation
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMalloc((void**)&d_ptr, 1024 * 1024));
    size_t free_after = 0, total_after = 0;
    CHECK_CUDA(UPTKMemGetInfo(&free_after, &total_after));
    printf("After alloc - Free: %zu, Total: %zu\n", free_after, total_after);
    CHECK_CUDA(UPTKFree(d_ptr));

    printf("test_cudaMemGetInfo PASS\n");
    return 0;
}
