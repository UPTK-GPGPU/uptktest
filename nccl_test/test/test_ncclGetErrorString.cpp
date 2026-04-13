#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_nccl.h>
#include <nccl.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    UPTKError_t cudaErr = UPTKGetDeviceCount(&deviceCount);
    if (cudaErr != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // 场景1: UPTKncclGetErrorString with UPTKncclSuccess
    {
        const char* str = UPTKncclGetErrorString(UPTKncclSuccess);
        if (str == NULL) {
            printf("UPTKncclGetErrorString(UPTKncclSuccess) returned NULL\n");
            return 1;
        }
        printf("UPTKncclSuccess: %s\n", str);
    }

    // 场景2: UPTKncclGetErrorString with UPTKncclUnhandledCudaError
    {
        const char* str = UPTKncclGetErrorString(UPTKncclUnhandledCudaError);
        if (str == NULL) {
            printf("UPTKncclGetErrorString(UPTKncclUnhandledCudaError) returned NULL\n");
            return 1;
        }
        printf("UPTKncclUnhandledCudaError: %s\n", str);
    }

    // 场景3: UPTKncclGetErrorString with UPTKncclInvalidArgument
    {
        const char* str = UPTKncclGetErrorString(UPTKncclInvalidArgument);
        if (str == NULL) {
            printf("UPTKncclGetErrorString(UPTKncclInvalidArgument) returned NULL\n");
            return 1;
        }
        printf("UPTKncclInvalidArgument: %s\n", str);
    }

    // 场景4: UPTKncclGetErrorString with UPTKncclInternalError
    {
        const char* str = UPTKncclGetErrorString(UPTKncclInternalError);
        if (str == NULL) {
            printf("UPTKncclGetErrorString(UPTKncclInternalError) returned NULL\n");
            return 1;
        }
        printf("UPTKncclInternalError: %s\n", str);
    }

    printf("test_UPTKncclGetErrorString PASS\n");
    return 0;
}
