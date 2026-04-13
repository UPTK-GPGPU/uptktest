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

    // Scene 1: Basic D2Z transform
    int N = 256;
    UPTKfftHandle plan;
    CHECK_CUFFT(UPTKfftPlan1d(&plan, N, UPTKFFT_D2Z, 1));

    UPTKfftDoubleReal *d_idata;
    UPTKfftDoubleComplex *d_odata;
    UPTKMalloc((void**)&d_idata, N * sizeof(UPTKfftDoubleReal));
    UPTKMalloc((void**)&d_odata, (N / 2 + 1) * sizeof(UPTKfftDoubleComplex));

    CHECK_CUFFT(UPTKfftExecD2Z(plan, d_idata, d_odata));

    UPTKFree(d_idata);
    UPTKFree(d_odata);
    CHECK_CUFFT(UPTKfftDestroy(plan));

    // Scene 2: D2Z with different size
    int N2 = 128;
    UPTKfftHandle plan2;
    CHECK_CUFFT(UPTKfftPlan1d(&plan2, N2, UPTKFFT_D2Z, 1));

    UPTKfftDoubleReal *d_idata2;
    UPTKfftDoubleComplex *d_odata2;
    UPTKMalloc((void**)&d_idata2, N2 * sizeof(UPTKfftDoubleReal));
    UPTKMalloc((void**)&d_odata2, (N2 / 2 + 1) * sizeof(UPTKfftDoubleComplex));

    CHECK_CUFFT(UPTKfftExecD2Z(plan2, d_idata2, d_odata2));

    UPTKFree(d_idata2);
    UPTKFree(d_odata2);
    CHECK_CUFFT(UPTKfftDestroy(plan2));

    printf("test_cufftExecD2Z PASS\n");
    return 0;
}
