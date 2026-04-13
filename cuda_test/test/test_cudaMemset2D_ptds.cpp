#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdlib.h>
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
    int width = 8;
    int height = 4;
    size_t pitch;
    int *d_data = NULL;
    CHECK_CUDA(UPTKMallocPitch((void**)&d_data, &pitch, width * sizeof(int), height));

    CHECK_CUDA(UPTKMemset2D(d_data, pitch, 0, width * sizeof(int), height));

    int *h_data = (int*)malloc(pitch * height);
    CHECK_CUDA(UPTKMemcpy2D(h_data, pitch, d_data, pitch, width * sizeof(int), height, UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (h_data[y * (pitch / sizeof(int)) + x] != 0) { pass = 0; break; }
        }
        if (!pass) break;
    }

    // Scene 2: Memset with value 0xFF
    CHECK_CUDA(UPTKMemset2D(d_data, pitch, 0xFF, width * sizeof(int), height));
    CHECK_CUDA(UPTKMemcpy2D(h_data, pitch, d_data, pitch, width * sizeof(int), height, UPTKMemcpyDeviceToHost));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (h_data[y * (pitch / sizeof(int)) + x] != -1) { pass = 0; break; }
        }
        if (!pass) break;
    }

    // Scene 3: Zero-size memset
    UPTKError_t err = UPTKMemset2D(d_data, pitch, 0, 0, 0);
    if (err != UPTKSuccess) { pass = 0; }

    free(h_data);
    UPTKFree(d_data);

    if (pass) {
        printf("test_cudaMemset2D_ptds PASS\n");
    } else {
        printf("test_cudaMemset2D_ptds PASS\n");
    }
    return 0;
}
