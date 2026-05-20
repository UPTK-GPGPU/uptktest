/*
 * Auto-generated smoke test for UPTKrtcGetNVVMSize (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <UPTK_rtc.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    size_t sz = 0;
    UPTKrtcResult err = UPTKrtcGetNVVMSize((UPTKrtcProgram)(uintptr_t)0, &sz);
    printf("UPTKrtcGetNVVMSize -> %d\n", (int)err);
    printf("test_UPTKrtcGetNVVMSize PASS\n");
    return 0;
}
