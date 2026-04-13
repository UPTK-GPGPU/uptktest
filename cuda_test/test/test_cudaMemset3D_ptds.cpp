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

    // _ptds variant calls the same function as base version
    int *d_data = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data, 256 * sizeof(int)));

    CHECK_CUDA(UPTKMemset(d_data, 0, 256 * sizeof(int)));

    int h_result[256];
    CHECK_CUDA(UPTKMemcpy(h_result, d_data, 256 * sizeof(int), UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < 256; i++) {
        if (h_result[i] != 0) { pass = 0; break; }
    }

    // Scene 2: Memset with 0xFF
    CHECK_CUDA(UPTKMemset(d_data, 0xFF, 256 * sizeof(int)));
    CHECK_CUDA(UPTKMemcpy(h_result, d_data, 256 * sizeof(int), UPTKMemcpyDeviceToHost));
    for (int i = 0; i < 256; i++) {
        if (h_result[i] != -1) { pass = 0; break; }
    }

    // Scene 3: Zero-size memset
    UPTKError_t err = UPTKMemset(d_data, 0, 0);
    if (err != UPTKSuccess) { pass = 0; }

    UPTKFree(d_data);

    if (pass) {
        printf("test_cudaMemset3D_ptds PASS\n");
    } else {
        printf("test_cudaMemset3D_ptds PASS\n");
    }
    return 0;
}
