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

    // Scenario 1: Set shared mem config to 4-byte bank size
    CHECK_CUDA(UPTKDeviceSetSharedMemConfig(UPTKSharedMemBankSizeFourByte));
    enum UPTKSharedMemConfig config;
    CHECK_CUDA(UPTKDeviceGetSharedMemConfig(&config));
    printf("Shared mem config after FourByte: %d\n", (int)config);

    // Scenario 2: Set shared mem config to 8-byte bank size
    UPTKError_t err = UPTKDeviceSetSharedMemConfig(UPTKSharedMemBankSizeEightByte);
    if (err == UPTKSuccess) {
        CHECK_CUDA(UPTKDeviceGetSharedMemConfig(&config));
        printf("Shared mem config after EightByte: %d\n", (int)config);
    } else {
        printf("EightByte config not supported: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceSetSharedMemConfig PASS\n");
    return 0;
}
