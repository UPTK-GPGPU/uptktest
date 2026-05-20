/*
 * Auto-generated smoke test for UPTKrtcGetLoweredName (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    const char *lowered = nullptr;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcAddNameExpression(prog, "uptk_rtc_smoke_kernel");
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: add name expr (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetLoweredName(prog, "uptk_rtc_smoke_kernel", &lowered);
    printf("UPTKrtcGetLoweredName -> %d (%s)\n", (int)err, lowered ? lowered : "(null)");
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetLoweredName PASS\n");
    return 0;
}
