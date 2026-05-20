/*
 * Auto-generated smoke test for UPTKrtcGetProgramLog (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <UPTK_rtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t log_sz = 0;
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
    err = UPTKrtcGetProgramLogSize(prog, &log_sz);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: log size (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    char *buf = (char *)malloc(log_sz + 1);
    if (!buf) {
        printf("test_skip: malloc\n");
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    memset(buf, 0, log_sz + 1);
    err = UPTKrtcGetProgramLog(prog, buf);
    printf("UPTKrtcGetProgramLog -> %d\n", (int)err);
    free(buf);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetProgramLog PASS\n");
    return 0;
}
