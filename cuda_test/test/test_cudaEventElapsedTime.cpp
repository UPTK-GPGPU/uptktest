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
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: Basic elapsed time measurement
    UPTKEvent_t start, stop;
    CHECK_CUDA(UPTKEventCreate(&start));
    CHECK_CUDA(UPTKEventCreate(&stop));

    CHECK_CUDA(UPTKEventRecord(start, 0));
    CHECK_CUDA(UPTKEventRecord(stop, 0));
    CHECK_CUDA(UPTKEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(UPTKEventElapsedTime(&ms, start, stop));
    printf("Elapsed time: %f ms\n", ms);

    CHECK_CUDA(UPTKEventDestroy(start));
    CHECK_CUDA(UPTKEventDestroy(stop));

    // Scene 2: Elapsed time with actual GPU work
    UPTKEvent_t start2, stop2;
    CHECK_CUDA(UPTKEventCreate(&start2));
    CHECK_CUDA(UPTKEventCreate(&stop2));

    float *d_a = NULL;
    CHECK_CUDA(UPTKMalloc(&d_a, 1024 * sizeof(float)));

    CHECK_CUDA(UPTKEventRecord(start2, 0));
    CHECK_CUDA(UPTKMemset(d_a, 0, 1024 * sizeof(float)));
    CHECK_CUDA(UPTKEventRecord(stop2, 0));
    CHECK_CUDA(UPTKEventSynchronize(stop2));

    float ms2 = 0.0f;
    CHECK_CUDA(UPTKEventElapsedTime(&ms2, start2, stop2));
    printf("Elapsed time with work: %f ms\n", ms2);

    CHECK_CUDA(UPTKFree(d_a));
    CHECK_CUDA(UPTKEventDestroy(start2));
    CHECK_CUDA(UPTKEventDestroy(stop2));

    printf("test_cudaEventElapsedTime PASS\n");
    return 0;
}
