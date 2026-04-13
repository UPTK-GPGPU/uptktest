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
    CHECK_CUDA(UPTKGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Get current shared memory config
    enum UPTKSharedMemConfig config;
    CHECK_CUDA(UPTKDeviceGetSharedMemConfig(&config));
    printf("Current shared mem config: %d\n", (int)config);

    // Scenario 2: Set to 4-byte bank config, then get it back
    CHECK_CUDA(UPTKDeviceSetSharedMemConfig(UPTKSharedMemBankSizeFourByte));
    enum UPTKSharedMemConfig newConfig;
    CHECK_CUDA(UPTKDeviceGetSharedMemConfig(&newConfig));
    printf("Shared mem config after setting: %d\n", (int)newConfig);

    // Scenario 3: Set to 8-byte bank config, then get it back
    CHECK_CUDA(UPTKDeviceSetSharedMemConfig(UPTKSharedMemBankSizeEightByte));
    CHECK_CUDA(UPTKDeviceGetSharedMemConfig(&newConfig));
    printf("Shared mem config after setting to EightByte: %d\n", (int)newConfig);

    printf("test_cudaDeviceGetSharedMemConfig PASS\n");
    return 0;
}
