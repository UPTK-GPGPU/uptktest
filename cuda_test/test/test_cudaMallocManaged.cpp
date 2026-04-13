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

    // Scene 1: Basic UPTKMallocManaged with default flags
    float *d_ptr = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr, 1024 * sizeof(float)));
    for (int i = 0; i < 256; i++) d_ptr[i] = (float)i;
    CHECK_CUDA(UPTKFree(d_ptr));

    // Scene 2: UPTKMallocManaged with UPTKMemAttachGlobal flag
    int *d_ptr2 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr2, 512 * sizeof(int), UPTKMemAttachGlobal));
    for (int i = 0; i < 128; i++) d_ptr2[i] = i;
    CHECK_CUDA(UPTKFree(d_ptr2));

    // Scene 3: UPTKMallocManaged with UPTKMemAttachHost flag
    double *d_ptr3 = NULL;
    CHECK_CUDA(UPTKMallocManaged((void**)&d_ptr3, 256 * sizeof(double), UPTKMemAttachHost));
    for (int i = 0; i < 64; i++) d_ptr3[i] = (double)i;
    CHECK_CUDA(UPTKFree(d_ptr3));

    printf("test_cudaMallocManaged PASS\n");
    return 0;
}
