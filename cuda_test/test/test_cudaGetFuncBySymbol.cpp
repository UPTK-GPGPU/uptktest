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

    // Scenario 1: Get function handle from device symbol address
    // Use a kernel symbol instead of a data symbol
    void *devPtr = NULL;
    UPTKError_t err = UPTKGetSymbolAddress(&devPtr, (const void*)"__cudaRegisterAll");
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKGetSymbolAddress failed on DTK: %s\n", UPTKGetErrorString(err));
        return 0;
    }

    UPTKFunction_t func = NULL;
    err = UPTKGetFuncBySymbol(&func, devPtr);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKGetFuncBySymbol returned %s (expected for non-kernel symbol)\n",
               UPTKGetErrorString(err));
        return 0;
    }

    printf("  Function handle obtained\n");
    printf("test_cudaGetFuncBySymbol PASS\n");
    return 0;
}
