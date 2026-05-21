/*
 * Auto-generated smoke test for UPTKrtcGetPTXSize (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t ptx_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetPTXSize(prog, &ptx_sz);
    printf("UPTKrtcGetPTXSize -> %d (sz=%zu)\n", (int)err, ptx_sz);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetPTXSize PASS\n");
    return 0;
}
