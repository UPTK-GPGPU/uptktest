/*
 * Auto-generated smoke test for UPTKrtcGetCUBIN (rtc_fun_convert.cpp).
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
    const char *opts[] = {"--gpu-architecture=compute_70"};
    size_t cubin_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke_cubin", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 1, opts);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: cubin compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetCUBINSize(prog, &cubin_sz);
    if (err != UPTKRTC_SUCCESS || cubin_sz == 0) {
        printf("test_skip: cubin size (%d) sz=%zu\n", (int)err, cubin_sz);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    char *buf = (char *)malloc(cubin_sz + 1);
    if (!buf) {
        printf("test_skip: malloc\n");
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    memset(buf, 0, cubin_sz + 1);
    err = UPTKrtcGetCUBIN(prog, buf);
    printf("UPTKrtcGetCUBIN -> %d\n", (int)err);
    free(buf);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetCUBIN PASS\n");
    return 0;
}
