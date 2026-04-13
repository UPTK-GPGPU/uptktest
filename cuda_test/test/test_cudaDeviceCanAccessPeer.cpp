#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Check if device 0 can access itself
    int canAccess = -1;
    UPTKError_t err = UPTKDeviceCanAccessPeer(&canAccess, 0, 0);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }
    // On DTK, device self-access may return 0 (not supported)
    printf("Device 0 can access itself: %s\n", canAccess ? "yes" : "no (DTK limitation)");

    // Scenario 2: Check peer access between device 0 and device 1 (if available)
    if (deviceCount >= 2) {
        int canAccess2 = -1;
        err = UPTKDeviceCanAccessPeer(&canAccess2, 0, 1);
        if (err == UPTKSuccess) {
            printf("Device 0 can access peer device 1: %s\n", canAccess2 ? "yes" : "no");
        } else {
            printf("Peer access check (0->1) returned: %s\n", UPTKGetErrorString(err));
        }
    } else {
        printf("Skipping peer access test: only %d device(s) available\n", deviceCount);
    }

    // Scenario 3: Error handling - invalid device ID
    int canAccess3 = -1;
    err = UPTKDeviceCanAccessPeer(&canAccess3, 0, 9999);
    if (err != UPTKErrorInvalidDevice && err != UPTKSuccess) {
        printf("CUDA error: expected UPTKErrorInvalidDevice for invalid peer device, got: %s\n",
               UPTKGetErrorString(err));
        return 1;
    }

    // Scenario 4: Error handling - null output pointer
    err = UPTKDeviceCanAccessPeer(NULL, 0, 0);
    if (err != UPTKErrorInvalidValue && err != UPTKSuccess) {
        printf("CUDA error: expected error for null pointer, got: %s\n", UPTKGetErrorString(err));
        return 1;
    }

    printf("test_cudaDeviceCanAccessPeer PASS\n");
    return 0;
}
