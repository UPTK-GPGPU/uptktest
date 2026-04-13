#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <UPTK_fft.h>
#include <UPTK_fftXt.h>
#include <cufft.h>
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

    // Scene 1: Plan 1D C2C
    UPTKfftHandle plan;
    CHECK_CUFFT(UPTKfftPlan1d(&plan, 256, UPTKFFT_C2C, 1));
    CHECK_CUFFT(UPTKfftDestroy(plan));

    // Scene 2: Plan 1D R2C
    UPTKfftHandle plan2;
    CHECK_CUFFT(UPTKfftPlan1d(&plan2, 512, UPTKFFT_R2C, 1));
    CHECK_CUFFT(UPTKfftDestroy(plan2));

    // Scene 3: Plan 1D Z2Z with batch
    UPTKfftHandle plan3;
    CHECK_CUFFT(UPTKfftPlan1d(&plan3, 128, UPTKFFT_Z2Z, 5));
    CHECK_CUFFT(UPTKfftDestroy(plan3));

    printf("test_cufftPlan1d PASS\n");
    return 0;
}
