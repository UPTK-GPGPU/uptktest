#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_nccl.h>
#include <nccl.h>
#include <stdio.h>

#define CHECK_NCCL(call) \
    do { \
        UPTKncclResult_t res = call; \
        if (res != UPTKncclSuccess) { \
            printf("NCCL error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKncclGetErrorString(res)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount = 0;
    UPTKError_t cudaErr = UPTKGetDeviceCount(&deviceCount);
    if (cudaErr != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // 场景1: 基础 UPTKncclGetVersion
    {
        int version = 0;
        CHECK_NCCL(UPTKncclGetVersion(&version));
        if (version <= 0) {
            printf("UPTKncclGetVersion: expected positive version, got %d\n", version);
            return 1;
        }
        printf("NCCL version: %d\n", version);
    }

    printf("test_UPTKncclGetVersion PASS\n");
    return 0;
}
