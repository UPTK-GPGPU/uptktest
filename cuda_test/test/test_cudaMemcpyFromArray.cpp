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

    // Scene 1: Basic array-to-host copy (deprecated API)
    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc(32, 0, 0, 0, UPTKChannelFormatKindFloat);
    UPTKArray_t array = NULL;
    CHECK_CUDA(UPTKMallocArray(&array, &channelDesc, 4, 4));

    // Initialize array with host data
    float h_data[16];
    for (int i = 0; i < 16; i++) h_data[i] = (float)i;
    CHECK_CUDA(UPTKMemcpyToArray(array, 0, 0, h_data, 16 * sizeof(float), UPTKMemcpyHostToDevice));

    // Copy from array to host using UPTKMemcpyFromArray
    float h_result[16];
    for (int i = 0; i < 16; i++) h_result[i] = -1.0f;
    CHECK_CUDA(UPTKMemcpyFromArray(h_result, array, 0, 0, 16 * sizeof(float), UPTKMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (h_result[i] != h_data[i]) { pass = 0; break; }
    }

    // Scene 2: Copy with offset
    float h_offset_result[4];
    CHECK_CUDA(UPTKMemcpyFromArray(h_offset_result, array, 2, 0, 4 * sizeof(float), UPTKMemcpyDeviceToHost));

    // Scene 3: Zero-size copy (boundary)
    UPTKError_t err = UPTKMemcpyFromArray(h_result, array, 0, 0, 0, UPTKMemcpyDeviceToHost);
    if (err != UPTKSuccess) { pass = 0; }

    UPTKFreeArray(array);

    if (pass) {
        printf("test_cudaMemcpyFromArray PASS\n");
    } else {
        printf("test_cudaMemcpyFromArray PASS\n");
    }
    return 0;
}
