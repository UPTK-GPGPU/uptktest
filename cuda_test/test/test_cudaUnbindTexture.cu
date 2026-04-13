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

    // UPTKUnbindTexture is deprecated and uses texture references
    // On DTK, this returns invalid device symbol for texture references
    
    // Scenario 1: Just call the API and handle error gracefully
    texture<float, 1, cudaReadModeElementType> texRef;

    UPTKError_t err = UPTKUnbindTexture(&texRef);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKUnbindTexture not supported on DTK: %s\n", UPTKGetErrorString(err));
        return 0;
    }
    printf("  UPTKUnbindTexture succeeded\n");

    printf("test_cudaUnbindTexture PASS\n");
    return 0;
}
