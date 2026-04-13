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

    // Scenario 1: Basic - enable peer access with device 1 if available, then disable
    if (deviceCount >= 2) {
        int canAccess;
        CHECK_CUDA(UPTKDeviceCanAccessPeer(&canAccess, 0, 1));
        if (canAccess) {
            UPTKError_t err = UPTKDeviceEnablePeerAccess(1, 0);
            if (err == UPTKSuccess) {
                CHECK_CUDA(UPTKDeviceDisablePeerAccess(1));
            }
        } else {
            printf("test_skip: device 0 cannot access device 1\n");
            return 0;
        }
    } else {
        printf("test_skip: need at least 2 devices for peer access test\n");
        return 0;
    }

    printf("test_cudaDeviceDisablePeerAccess PASS\n");
    return 0;
}
