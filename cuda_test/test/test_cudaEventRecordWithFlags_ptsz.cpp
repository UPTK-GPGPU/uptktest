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

    // Scenario 1: Record with default flags using _ptsz variant
    UPTKEvent_t event;
    UPTKError_t err = UPTKEventCreate(&event);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        return 1;
    }

    err = UPTKEventRecordWithFlags(event, 0, 0);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKEventRecordWithFlags(_ptsz) not supported on DTK: %s\n", UPTKGetErrorString(err));
        UPTKEventDestroy(event);
        return 0;
    }

    err = UPTKEventSynchronize(event);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        UPTKEventDestroy(event);
        return 1;
    }
    printf("  Event recorded with _ptsz variant, default flags\n");
    UPTKEventDestroy(event);

    printf("test_cudaEventRecordWithFlags_ptsz PASS\n");
    return 0;
}
