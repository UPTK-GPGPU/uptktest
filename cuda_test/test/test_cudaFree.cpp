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

    // Scene 1: Basic malloc and free
    float *d_a = NULL;
    CHECK_CUDA(UPTKMalloc(&d_a, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKFree(d_a));
    printf("Basic malloc/free succeeded\n");

    // Scene 2: Free NULL pointer (should be safe)
    CHECK_CUDA(UPTKFree(NULL));
    printf("Free NULL pointer succeeded\n");

    // Scene 3: Multiple allocations and frees
    float *d_b = NULL;
    int *d_c = NULL;
    CHECK_CUDA(UPTKMalloc(&d_b, 2048 * sizeof(float)));
    CHECK_CUDA(UPTKMalloc(&d_c, 512 * sizeof(int)));
    CHECK_CUDA(UPTKFree(d_b));
    CHECK_CUDA(UPTKFree(d_c));
    printf("Multiple malloc/free succeeded\n");

    printf("test_cudaFree PASS\n");
    return 0;
}
