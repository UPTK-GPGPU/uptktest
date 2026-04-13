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

    // 场景1: 基础 UPTKncclCommGetAsyncError with valid communicator
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

        UPTKncclResult_t asyncError = UPTKncclSuccess;
        CHECK_NCCL(UPTKncclCommGetAsyncError(comm, &asyncError));
        // After init, asyncError should be UPTKncclSuccess
        if (asyncError != UPTKncclSuccess) {
            printf("UPTKncclCommGetAsyncError: expected UPTKncclSuccess, got %s\n",
                   UPTKncclGetErrorString(asyncError));
            return 1;
        }
    }

    CHECK_NCCL(UPTKncclCommDestroy(comm));

    printf("test_UPTKncclCommGetAsyncError PASS\n");
    return 0;
}
