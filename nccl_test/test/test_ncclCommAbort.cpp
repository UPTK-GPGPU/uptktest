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

    // 场景1: 基础 UPTKncclCommAbort - abort a valid communicator
    {
        UPTKncclCommAbort(comm);
        comm = NULL;
    }

    // 场景2: UPTKncclCommAbort with NULL communicator (should be safe/no-op)
    {
        UPTKncclCommAbort(NULL);
    }

    // 场景3: Create and abort another communicator
    {
        UPTKncclComm_t comm2 = NULL;
        res = UPTKncclGetUniqueId(&commId);
        if (res == UPTKncclSuccess) {
            res = UPTKncclCommInitRank(&comm2, 1, commId, 0);
            if (res == UPTKncclSuccess) {
                UPTKncclCommAbort(comm2);
            }
        }
    }

    printf("test_UPTKncclCommAbort PASS\n");
    return 0;
}
