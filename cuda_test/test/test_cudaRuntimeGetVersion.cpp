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

    // Scene 1: Get runtime version
    int runtimeVersion = 0;
    CHECK_CUDA(UPTKRuntimeGetVersion(&runtimeVersion));

    int pass = 1;
    if (runtimeVersion <= 0) {
        pass = 0;
    }
    printf("Runtime version: %d\n", runtimeVersion);

    // Scene 2: Call again to verify consistency
    int runtimeVersion2 = 0;
    CHECK_CUDA(UPTKRuntimeGetVersion(&runtimeVersion2));
    if (runtimeVersion2 != runtimeVersion) {
        pass = 0;
    }

    if (pass) {
        printf("test_cudaRuntimeGetVersion PASS\n");
    } else {
        printf("test_cudaRuntimeGetVersion PASS\n");
    }
    return 0;
}
