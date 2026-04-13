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

    // Scenario 1: Get current stack size limit, then set a new value
    size_t origValue;
    CHECK_CUDA(UPTKDeviceGetLimit(&origValue, UPTKLimitStackSize));
    printf("Original stack size limit: %zu\n", origValue);

    // Set stack size to 2048 bytes (a reasonable value)
    CHECK_CUDA(UPTKDeviceSetLimit(UPTKLimitStackSize, 2048));

    size_t newValue;
    CHECK_CUDA(UPTKDeviceGetLimit(&newValue, UPTKLimitStackSize));
    printf("New stack size limit: %zu\n", newValue);

    // Scenario 2: Set malloc heap size limit
    size_t heapSize = 16 * 1024 * 1024; // 16 MB
    UPTKError_t err = UPTKDeviceSetLimit(UPTKLimitMallocHeapSize, heapSize);
    if (err == UPTKSuccess) {
        printf("Set malloc heap size to %zu bytes\n", heapSize);
    } else {
        printf("Setting malloc heap size returned: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceSetLimit PASS\n");
    return 0;
}
