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

    // Note: UPTKImportExternalMemory requires a valid external memory handle
    // from another API (Vulkan, D3D12, etc.). On DTK/AMD GPU, passing invalid
    // or NULL parameters causes an abort rather than returning an error code.
    // Without a valid external memory handle, this API cannot be safely tested.
    printf("test_skip: UPTKImportExternalMemory requires valid external memory handle from Vulkan/D3D12, not available in this environment\n");
    return 0;
}
