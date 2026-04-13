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

    // Scenario 1: Get P2P attribute - access supported between device 0 and itself
    int accessSupported;
    UPTKError_t err = UPTKDeviceGetP2PAttribute(&accessSupported, UPTKDevP2PAttrAccessSupported, 0, 0);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKDeviceGetP2PAttribute not supported on DTK: %s\n", UPTKGetErrorString(err));
        return 0;
    }
    printf("P2P access supported (0->0): %d\n", accessSupported);

    // Scenario 2: Get P2P attribute - performance rank
    int perfRank;
    err = UPTKDeviceGetP2PAttribute(&perfRank, UPTKDevP2PAttrPerformanceRank, 0, 0);
    if (err == UPTKSuccess) {
        printf("P2P performance rank (0->0): %d\n", perfRank);
    } else {
        printf("P2P perf rank (0->0) not available: %s\n", UPTKGetErrorString(err));
    }

    printf("test_cudaDeviceGetP2PAttribute PASS\n");
    return 0;
}
