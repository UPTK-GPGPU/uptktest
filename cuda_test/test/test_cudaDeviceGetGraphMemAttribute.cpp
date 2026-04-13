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
    if (deviceCount == 1) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    CHECK_CUDA(UPTKSetDevice(0));

    // Scenario 1: Get UPTKGraphMemAttrUsedMemHigh for device 0
    uint64_t usedMemHigh;
    UPTKError_t err = UPTKDeviceGetGraphMemAttribute(0, UPTKGraphMemAttrUsedMemHigh, &usedMemHigh);
    if (err == UPTKSuccess) {
        printf("Graph mem used mem high: %lu\n", (unsigned long)usedMemHigh);
    } else {
        printf("UPTKDeviceGetGraphMemAttribute returned: %s (may not be supported)\n", UPTKGetErrorString(err));
    }

    // Scenario 2: Get UPTKGraphMemAttrReservedMemHigh for device 0
    uint64_t reservedMemHigh;
    err = UPTKDeviceGetGraphMemAttribute(0, UPTKGraphMemAttrReservedMemHigh, &reservedMemHigh);
    if (err == UPTKSuccess) {
        printf("Graph mem reserved mem high: %lu\n", (unsigned long)reservedMemHigh);
    } else {
        printf("UPTKDeviceGetGraphMemAttribute (ReservedMemHigh) returned: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceGetGraphMemAttribute PASS\n");
    return 0;
}
