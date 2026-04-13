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

    // Scenario 1: Get PCI bus ID for device 0
    char pciBusId[13];
    CHECK_CUDA(UPTKDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), 0));
    printf("Device 0 PCI Bus ID: %s\n", pciBusId);

    // Scenario 2: Get PCI bus ID for device 0 with larger buffer
    char pciBusIdLarge[64];
    CHECK_CUDA(UPTKDeviceGetPCIBusId(pciBusIdLarge, sizeof(pciBusIdLarge), 0));
    printf("Device 0 PCI Bus ID (large buffer): %s\n", pciBusIdLarge);

    // Scenario 3: If multiple devices, get PCI bus ID for device 1
    if (deviceCount >= 2) {
        char pciBusId1[13];
        CHECK_CUDA(UPTKDeviceGetPCIBusId(pciBusId1, sizeof(pciBusId1), 1));
        printf("Device 1 PCI Bus ID: %s\n", pciBusId1);
    }

    printf("test_cudaDeviceGetPCIBusId PASS\n");
    return 0;
}
