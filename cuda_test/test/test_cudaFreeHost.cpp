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

    // Scene 1: Basic host memory alloc and free
    float *h_a = NULL;
    CHECK_CUDA(UPTKMallocHost(&h_a, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKFreeHost(h_a));
    printf("Basic host malloc/free succeeded\n");

    // Scene 2: Free NULL host pointer (should be safe)
    CHECK_CUDA(UPTKFreeHost(NULL));
    printf("Free NULL host pointer succeeded\n");

    // Scene 3: Multiple host allocations and frees
    int *h_b = NULL;
    double *h_c = NULL;
    CHECK_CUDA(UPTKMallocHost(&h_b, 512 * sizeof(int)));
    CHECK_CUDA(UPTKMallocHost(&h_c, 256 * sizeof(double)));
    CHECK_CUDA(UPTKFreeHost(h_b));
    CHECK_CUDA(UPTKFreeHost(h_c));
    printf("Multiple host malloc/free succeeded\n");

    printf("test_cudaFreeHost PASS\n");
    return 0;
}
