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

    // 场景1: 基础 UPTKncclGetUniqueId
    {
        UPTKncclUniqueId uniqueId;
        CHECK_NCCL(UPTKncclGetUniqueId(&uniqueId));
        // Verify the uniqueId is not all zeros
        int all_zero = 1;
        for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
            if (uniqueId.internal[i] != 0) {
                all_zero = 0;
                break;
            }
        }
        if (all_zero) {
            printf("UPTKncclGetUniqueId: uniqueId is all zeros\n");
            return 1;
        }
    }

    // 场景2: Generate multiple uniqueIds and verify they are different
    {
        UPTKncclUniqueId id1, id2;
        CHECK_NCCL(UPTKncclGetUniqueId(&id1));
        CHECK_NCCL(UPTKncclGetUniqueId(&id2));

        int same = 1;
        for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
            if (id1.internal[i] != id2.internal[i]) {
                same = 0;
                break;
            }
        }
        if (same) {
            printf("UPTKncclGetUniqueId: two consecutive calls returned same ID\n");
            return 1;
        }
    }

    printf("test_UPTKncclGetUniqueId PASS\n");
    return 0;
}
