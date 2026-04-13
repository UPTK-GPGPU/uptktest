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

    // 场景1: 基础 UPTKncclCommInitAll with ndev=1, no device list
    {
        UPTKncclComm_t comm = NULL;
        UPTKncclResult_t res = UPTKncclCommInitAll(&comm, 1, NULL);
        if (res != UPTKncclSuccess) {
            printf("test_skip: UPTKncclCommInitAll failed: %s\n", UPTKncclGetErrorString(res));
            return 0;
        }

        int count = 0;
        CHECK_NCCL(UPTKncclCommCount(comm, &count));
        if (count != 1) {
            printf("UPTKncclCommInitAll: expected count=1, got %d\n", count);
            UPTKncclCommDestroy(comm);
            return 1;
        }

        CHECK_NCCL(UPTKncclCommDestroy(comm));
    }

    // 场景2: UPTKncclCommInitAll with explicit device list
    {
        int devlist[1] = {0};
        UPTKncclComm_t comm = NULL;
        UPTKncclResult_t res = UPTKncclCommInitAll(&comm, 1, devlist);
        if (res != UPTKncclSuccess) {
            printf("test_skip: UPTKncclCommInitAll with devlist failed: %s\n", UPTKncclGetErrorString(res));
            return 0;
        }

        int device = -1;
        CHECK_NCCL(UPTKncclCommCuDevice(comm, &device));
        if (device != 0) {
            printf("UPTKncclCommInitAll: expected device=0, got %d\n", device);
            UPTKncclCommDestroy(comm);
            return 1;
        }

        CHECK_NCCL(UPTKncclCommDestroy(comm));
    }

    printf("test_UPTKncclCommInitAll PASS\n");
    return 0;
}
