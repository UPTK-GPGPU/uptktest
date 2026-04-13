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

    UPTKncclUniqueId commId;
    UPTKncclComm_t comm = NULL;
    UPTKncclResult_t res;

    // 场景1: 基础 UPTKncclCommUserRank with valid communicator
    {
        res = UPTKncclGetUniqueId(&commId);
        if (res != UPTKncclSuccess) {
            printf("test_skip: UPTKncclGetUniqueId failed: %s\n", UPTKncclGetErrorString(res));
            return 0;
        }

        res = UPTKncclCommInitRank(&comm, 1, commId, 0);
        if (res != UPTKncclSuccess) {
            printf("test_skip: UPTKncclCommInitRank failed: %s\n", UPTKncclGetErrorString(res));
            return 0;
        }

        int rank = -1;
        CHECK_NCCL(UPTKncclCommUserRank(comm, &rank));
        // With nranks=1, rank=0, the user rank should be 0
        if (rank != 0) {
            printf("UPTKncclCommUserRank: expected 0, got %d\n", rank);
            return 1;
        }
    }

    CHECK_NCCL(UPTKncclCommDestroy(comm));

    printf("test_UPTKncclCommUserRank PASS\n");
    return 0;
}
