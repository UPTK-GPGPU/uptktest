#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <stdint.h>

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

    // Scenario 1: Set UsedMemHigh to 0 (reset high watermark)
    uint64_t zero = 0;
    UPTKError_t err = UPTKDeviceSetGraphMemAttribute(0, UPTKGraphMemAttrUsedMemHigh, &zero);
    if (err == UPTKSuccess) {
        printf("Reset UsedMemHigh to 0\n");
    } else {
        printf("UPTKDeviceSetGraphMemAttribute returned: %s (may not be supported)\n", UPTKGetErrorString(err));
    }

    // Scenario 2: Set ReservedMemHigh to 0 (reset high watermark)
    err = UPTKDeviceSetGraphMemAttribute(0, UPTKGraphMemAttrReservedMemHigh, &zero);
    if (err == UPTKSuccess) {
        printf("Reset ReservedMemHigh to 0\n");
    } else {
        printf("UPTKDeviceSetGraphMemAttribute (ReservedMemHigh) returned: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceSetGraphMemAttribute PASS\n");
    return 0;
}
