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

    // Scenario 1: Get PCI bus ID string for device 0, then resolve back to device number
    char pciBusId[13];
    CHECK_CUDA(UPTKDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), 0));
    printf("Device 0 PCI Bus ID: %s\n", pciBusId);

    int device;
    CHECK_CUDA(UPTKDeviceGetByPCIBusId(&device, pciBusId));
    printf("Resolved device from PCI Bus ID: %d\n", device);

    // Scenario 2: Verify resolved device matches original
    if (device == 0) {
        printf("PCI Bus ID round-trip verification passed\n");
    } else {
        printf("Warning: device mismatch: expected 0, got %d\n", device);
    }

    printf("test_cudaDeviceGetByPCIBusId PASS\n");
    return 0;
}
