#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
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
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic H2D + D2H async copy
    int h_data[256];
    int h_result[256];
    for (int i = 0; i < 256; i++) h_data[i] = i;

    int *d_data = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data, 256 * sizeof(int)));

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    CHECK_CUDA(UPTKMemcpyAsync(d_data, h_data, 256 * sizeof(int), UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKMemcpyAsync(h_result, d_data, 256 * sizeof(int), UPTKMemcpyDeviceToHost, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    int pass = 1;
    for (int i = 0; i < 256; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: D2D async copy
    int *d_data2 = NULL;
    CHECK_CUDA(UPTKMalloc(&d_data2, 256 * sizeof(int)));
    CHECK_CUDA(UPTKMemcpyAsync(d_data2, d_data, 256 * sizeof(int), UPTKMemcpyDeviceToDevice, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Zero-size copy (boundary condition)
    UPTKError_t err = UPTKMemcpyAsync(d_data, d_data2, 0, UPTKMemcpyDeviceToDevice, stream);
    if (err != UPTKSuccess) { pass = 0; }

    // Scene 4: NULL stream (default stream)
    CHECK_CUDA(UPTKMemcpyAsync(h_result, d_data, 256 * sizeof(int), UPTKMemcpyDeviceToHost, 0));
    CHECK_CUDA(UPTKDeviceSynchronize());

    UPTKFree(d_data);
    UPTKFree(d_data2);
    UPTKStreamDestroy(stream);

    printf("test_cudaMemcpyAsync PASS\n");
    return 0;
}
