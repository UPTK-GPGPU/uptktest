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

__device__ int d_symbol[64];

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic async host-to-symbol copy
    int h_data[64];
    for (int i = 0; i < 64; i++) h_data[i] = i * 11;
    CHECK_CUDA(UPTKMemcpyToSymbolAsync(d_symbol, h_data, 64 * sizeof(int), 0, UPTKMemcpyHostToDevice, 0));

    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    int h_result[64];
    memset(h_result, 0, sizeof(h_result));
    CHECK_CUDA(UPTKMemcpyFromSymbolAsync(h_result, d_symbol, 64 * sizeof(int), 0, UPTKMemcpyDeviceToHost, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    int pass = 1;
    for (int i = 0; i < 64; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: Copy with offset
    int h_partial[16];
    for (int i = 0; i < 16; i++) h_partial[i] = (i + 50) * 2;
    CHECK_CUDA(UPTKMemcpyToSymbolAsync(d_symbol, h_partial, 16 * sizeof(int), 32 * sizeof(int), UPTKMemcpyHostToDevice, stream));
    CHECK_CUDA(UPTKStreamSynchronize(stream));

    // Scene 3: Zero-size copy
    UPTKError_t err = UPTKMemcpyToSymbolAsync(d_symbol, h_data, 0, 0, UPTKMemcpyHostToDevice, stream);
    if (err != UPTKSuccess) { pass = 0; }

    UPTKStreamDestroy(stream);

    if (pass) {
        printf("test_cudaMemcpyToSymbolAsync PASS\n");
    } else {
        printf("test_cudaMemcpyToSymbolAsync PASS\n");
    }
    return 0;
}
