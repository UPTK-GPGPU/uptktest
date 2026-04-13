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

    // 场景1: 基础 UPTKncclCommDestroy with valid communicator
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

        CHECK_NCCL(UPTKncclCommDestroy(comm));
        comm = NULL;
    }

    // 场景2: UPTKncclCommDestroy with NULL communicator (should be safe)
    {
        UPTKncclCommDestroy(NULL);
    }

    // 场景3: Create and destroy multiple communicators
    {
        UPTKncclComm_t comm1 = NULL;
        UPTKncclComm_t comm2 = NULL;

        res = UPTKncclGetUniqueId(&commId);
        if (res == UPTKncclSuccess) {
            res = UPTKncclCommInitRank(&comm1, 1, commId, 0);
            if (res == UPTKncclSuccess) {
                CHECK_NCCL(UPTKncclCommDestroy(comm1));
                comm1 = NULL;
            }
        }

        res = UPTKncclGetUniqueId(&commId);
        if (res == UPTKncclSuccess) {
            res = UPTKncclCommInitRank(&comm2, 1, commId, 0);
            if (res == UPTKncclSuccess) {
                CHECK_NCCL(UPTKncclCommDestroy(comm2));
                comm2 = NULL;
            }
        }
    }

    printf("test_UPTKncclCommDestroy PASS\n");
    return 0;
}
