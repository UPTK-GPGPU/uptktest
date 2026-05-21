/*
 * Auto-generated smoke test for UPTKrtcGetNVVM (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rtc.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    char tiny[4];
    tiny[0] = 0;
    UPTKrtcResult err = UPTKrtcGetNVVM((UPTKrtcProgram)(uintptr_t)0, tiny);
    printf("UPTKrtcGetNVVM -> %d\n", (int)err);
    printf("test_UPTKrtcGetNVVM PASS\n");
    return 0;
}
