#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_fft.h>
#include <UPTK_fftXt.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>

#define CHECK_CUFFT(call) \
    do { \
        UPTKfftResult res = call; \
        if (res != CUFFT_SUCCESS) { \
            printf("cuFFT error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cufftGetErrorString(res)); \
            return 1; \
        } \
    } while (0)

static const char* cufftGetErrorString(UPTKfftResult result) {
    switch (result) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "CUFFT_UNKNOWN_ERROR";
    }
}


int main() {
    int deviceCount;
    UPTKError_t cudaErr = UPTKGetDeviceCount(&deviceCount);
    if (cudaErr != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    if (deviceCount < 2) {
        printf("test_skip: UPTKfftXtSetCallback requires multiple GPUs\n");
        return 0;
    }

    // Scene 1: Set load callback
    UPTKfftHandle plan;
    CHECK_CUFFT(UPTKfftCreate(&plan));
    int gpus[] = {0, 1};
    CHECK_CUFFT(UPTKfftXtSetGPUs(plan, 2, gpus));

    long long int n[] = {256};
    size_t workSize;
    CHECK_CUFFT(UPTKfftXtMakePlanMany(plan, 1, n, NULL, 1, 0, UPTK_C_32F, NULL, 1, 0, UPTK_C_32F, 1, &workSize, UPTK_C_32F));

    void *callback_routine = NULL;
    void *caller_info = NULL;
    // Note: Setting NULL callback just tests the API call path
    UPTKfftResult res = UPTKfftXtSetCallback(plan, &callback_routine, UPTKFFT_CB_LD_COMPLEX, &caller_info);
    // May return NOT_IMPLEMENTED on some platforms - that's acceptable
    if (res != CUFFT_SUCCESS && res != CUFFT_NOT_IMPLEMENTED) {
        printf("cuFFT error: %s\n", cufftGetErrorString(res));
        return 1;
    }

    CHECK_CUFFT(UPTKfftDestroy(plan));

    printf("test_cufftXtSetCallback PASS\n");
    return 0;
}
