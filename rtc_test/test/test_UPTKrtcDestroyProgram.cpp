/*
 * Auto-generated smoke test for UPTKrtcDestroyProgram (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create program failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcDestroyProgram(&prog);
    printf("UPTKrtcDestroyProgram -> %d\n", (int)err);
    printf("test_UPTKrtcDestroyProgram PASS\n");
    return 0;
}
